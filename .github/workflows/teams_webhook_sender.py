#!/usr/bin/env python3
"""Post nightly build status to a Teams Workflow webhook (Power Automate).

Legacy Office 365 Connectors accepted {"title", "text"}. Microsoft retired those
connectors; "Send webhook alerts to a channel" expects an Adaptive Card inside
a message envelope. See:
https://learn.microsoft.com/en-us/microsoftteams/platform/webhooks-and-connectors/how-to/add-incoming-webhook
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

def _parse_markdown_table(md_text: str) -> list[list[str]] | None:
    """Return a list of rows (each a list of cell strings) or *None*"""

    lines = [ln.strip() for ln in md_text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    rows: list[list[str]] = []
    for line in lines:
        if re.match(r"^\|[\s\-:|]+\|$", line):
            continue
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    return rows if len(rows) >= 2 else None  # header + at least one data row


def _adaptive_card_table(rows: list[list[str]], accent_header: bool = True) -> dict:
    """Build an Adaptive Card *Table* element (schema ≥ 1.5)"""
    header, *data_rows = rows
    num_cols = len(header)
    columns = [{"width": 1} for _ in range(num_cols)]

    def _make_row(cells: list[str], *, style: str | None = None, is_header: bool = False) -> dict:
        row: dict = {
            "type": "TableRow",
            "cells": [
                {
                    "type": "TableCell",
                    "items": [
                        {
                            "type": "TextBlock",
                            "text": cells[i] if i < len(cells) else "",
                            "size": "Small",
                            "spacing": "None",
                            "wrap": True,
                            **({"weight": "Bolder"} if is_header else {}),
                        }
                    ],
                }
                for i in range(num_cols)
            ],
        }
        if style:
            row["style"] = style
        return row

    table_rows = [_make_row(header, style="Accent" if accent_header else None, is_header=True)]
    for dr in data_rows:
        table_rows.append(_make_row(dr))

    return {
        "type": "Table",
        "gridStyle": "Default",
        "firstRowAsHeader": True,
        "showGridLines": True,
        "spacing": "Small",
        "columns": columns,
        "rows": table_rows,
    }

def normalize_table_text(text: str) -> str:
    return text.replace("\\n", "\n")


def _table_or_textblock(md_text: str, fallback_label: str = "") -> list[dict]:
    """Try to convert *md_text* to a Table element; fall back to TextBlock."""
    if not md_text or md_text.strip().lower() == "no failures found":
        return []

    parsed = _parse_markdown_table(md_text)
    if parsed:
        return [_adaptive_card_table(parsed)]

    # Not a recognisable table – render as plain text
    return [
        {
            "type": "TextBlock",
            "text": md_text,
            "wrap": True,
            "fontType": "Monospace",
        }
    ]


def build_adaptive_card_payload(title: str, *, body_elements: list[dict]) -> dict:
    return {
        "type": "message",
        "summary": title,
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "contentUrl": None,
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.5",
                    "msteams": {"width": "Full"},
                    "body": body_elements,
                },
            }
        ],
    }


def infer_status(results: dict | None) -> str:
    if not results:
        return "failure"
    failure_table = results.get("failure_table", "")
    if failure_table and failure_table != "No failures found":
        return "failure"
    return "success"


def build_body_elements(
    *,
    repository: str,
    branch: str,
    status: str,
    run_url: str,
    results: dict | None,
) -> list[dict]:
    """Return a list of Adaptive Card body elements."""

    icon = "✅" if status == "success" else "❌"

    elements: list[dict] = [
        # ---- metadata fact set (compact key/value pairs) ----
        {
            "type": "FactSet",
            "facts": [
                {"title": "Repository", "value": repository},
                {"title": "Branch", "value": branch},
                {"title": "Status", "value": f"{icon} {status}"},
            ],
        },
        # ---- build link ----
        {
            "type": "TextBlock",
            "text": f"🔗 [Build Link]({run_url})",
            "wrap": True,
        },
    ]

    if results:
        # -- Manifest / Artifacts table --
        manifest_md = normalize_table_text(results.get("manifest_artifacts_table", ""))
        if manifest_md:
            elements.append(
                {
                    "type": "TextBlock",
                    "text": "**Manifest, Artifacts, and Logs**",
                    "weight": "Bolder",
                    "size": "Small",
                    "spacing": "None",
                    "wrap": True,
                }
            )
            elements.extend(_table_or_textblock(manifest_md, "Manifest"))

        # -- Submodule table --
        submodule_md = normalize_table_text(results.get("submodule_table", ""))
        if submodule_md:
            elements.append(
                {
                    "type": "TextBlock",
                    "text": "**Details**",
                    "weight": "Bolder",
                    "size": "Small",
                    "spacing": "None",
                    "wrap": True,
                }
            )
            elements.extend(_table_or_textblock(submodule_md, "Submodules"))

        # -- Failure table --
        failure_md = normalize_table_text(results.get("failure_table", ""))
        if failure_md and failure_md.strip().lower() != "no failures found":
            elements.append(
                {
                    "type": "TextBlock",
                    "text": "**Failure Jobs**",
                    "weight": "Bolder",
                    "size": "Small",
                    "spacing": "None",
                    "wrap": True,
                    "color": "Attention",
                }
            )
            elements.extend(_table_or_textblock(failure_md, "Failures"))
    else:
        elements.append(
            {
                "type": "TextBlock",
                "text": "⚠️ Unable to fetch detailed results. Please check logs.",
                "wrap": True,
                "color": "Warning",
            }
        )

    return elements

def post_payload(webhook_url: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8", errors="replace")
            print(f"Teams webhook HTTP {response.status}")
            if body.strip():
                print(body)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"Teams webhook HTTP {exc.code}", file=sys.stderr)
        if error_body.strip():
            print(error_body, file=sys.stderr)
        raise SystemExit(1) from exc

def main() -> None:
    webhook_url = os.environ["TEAMS_WEBHOOK_URL"]
    results_path = os.environ.get("JSON_RESULT_FILE", "results.json")
    repository = os.environ["GITHUB_REPOSITORY"]
    branch = os.environ.get("GITHUB_REF_NAME", os.environ.get("GITHUB_REF", ""))
    run_id = os.environ["RUN_ID"]
    tz_name = os.environ.get("TZ", "America/Chicago")

    run_url = f"https://github.com/{repository}/actions/runs/{run_id}"
    start_date = datetime.now(ZoneInfo(tz_name)).strftime("%Y-%m-%d")

    results = None
    if os.path.isfile(results_path):
        with open(results_path, encoding="utf-8") as handle:
            results = json.load(handle)

    status = infer_status(results)
    icon = "✅" if status == "success" else "❌"
    title = f"{start_date} - {icon} Build {status}"

    body_elements = [
        {
            "type": "TextBlock",
            "text": title,
            "weight": "Bolder",
            "size": "ExtraLarge",
            "wrap": True,
        },
    ] + build_body_elements(
        repository=repository,
        branch=branch,
        status=status,
        run_url=run_url,
        results=results,
    )

    payload = build_adaptive_card_payload(title, body_elements=body_elements)
    print("Notification payload:")
    print(json.dumps(payload, indent=2))
    post_payload(webhook_url, payload)


if __name__ == "__main__":
    main()
