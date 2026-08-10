#!/usr/bin/env python3
# A tool to parse the output of `llvm-bolt --help-hidden` and update the
# documentation in CommandLineArgumentReference.md automatically.
# Run from the directory in which this file is located to update the docs.

import subprocess
from textwrap import wrap

LINE_LIMIT = 80


def wrap_text(text, indent, limit=LINE_LIMIT):
    wrapped_lines = wrap(text, width=limit - len(indent))
    wrapped_text = ("\n" + indent).join(wrapped_lines)
    return wrapped_text


def add_info(sections, section, option, description):
    indent = "  "
    wrapped_description = "\n".join(
        [
            wrap_text(line, indent) if len(line) > LINE_LIMIT else line
            for line in description
        ]
    )
    sections[section].append((option, indent + wrapped_description))


def parse_bolt_options(output):
    section_headers = [
        "Generic options:",
        "Output options:",
        "BOLT generic options:",
        "BOLT optimization options:",
        "BOLT options in relocation mode:",
        "BOLT instrumentation options:",
        "BOLT printing options:",
    ]

    sections = {key: [] for key in section_headers}
    current_section, prev_section = None, None
    option, description = None, []

    for line in output.split("\n"):
        cleaned_line = line.strip()

        if cleaned_line.casefold() in map(str.casefold, section_headers):
            if prev_section is not None:  # Save last option from prev section
                add_info(sections, current_section, option, description)
                option, description = None, []

            cleaned_line = cleaned_line.split()
            # Apply lowercase to all words except the first one
            cleaned_line = [cleaned_line[0]] + [
                word.lower() for word in cleaned_line[1:]
            ]
            # Join the words back together into a string
            cleaned_line = " ".join(cleaned_line)

            current_section = cleaned_line
            prev_section = current_section
            continue

        if cleaned_line.startswith("-"):
            if option and description:
                # Join description lines, adding an extra newline for
                # sub-options that start with '='
                add_info(sections, current_section, option, description)
                option, description = None, []

            parts = cleaned_line.split("  ", 1)
            if len(parts) > 1:
                option = parts[0].strip()
                descr = parts[1].strip()
                descr = descr[2].upper() + descr[3:]
                description = [descr]
                if option.startswith("--print") or option.startswith("--time"):
                    current_section = "BOLT printing options:"
                elif prev_section is not None:
                    current_section = prev_section
            continue

        if cleaned_line.startswith("="):
            parts = cleaned_line.split(maxsplit=1)
            # Split into two parts: sub-option and description
            if len(parts) == 2:
                # Rejoin with a single space
                cleaned_line = parts[0] + " " + parts[1].rstrip()
            description.append(cleaned_line)
        elif cleaned_line:  # Multiline description continuation
            description.append(cleaned_line)

    add_info(sections, current_section, option, description)
    return sections


def generate_markdown(sections):
    markdown_lines = [
        "# BOLT - a post-link optimizer developed to speed up large applications\n",
        "## SYNOPSIS\n",
        "`llvm-bolt <executable> [-o outputfile] <executable>.bolt "
        "[-data=perf.fdata] [options]`\n",
        "## OPTIONS",
    ]

    for section, options in sections.items():
        markdown_lines.append(f"\n### {section}")
        if section == "BOLT instrumentation options:":
            markdown_lines.append(
                f"\n`llvm-bolt <executable> -instrument"
                " [-o outputfile] <instrumented-executable>`"
            )
        for option, desc in options:
            markdown_lines.append(f"\n- `{option}`\n")
            # Split description into lines to handle sub-options
            desc_lines = desc.split("\n")
            for line in desc_lines:
                if line.startswith("="):
                    # Sub-option: correct formatting with bullet
                    sub_option, sub_desc = line[1:].split(" ", 1)
                    markdown_lines.append(f"  - `{sub_option}`: {sub_desc[4:]}")
                else:
                    # Regular line of description
                    if line[2:].startswith("<"):
                        line = line.replace("<", "").replace(">", "")
                    markdown_lines.append(f"{line}")

    return "\n".join(markdown_lines)


def main():
    try:
        help_output = subprocess.run(
            ["llvm-bolt", "--help-hidden"], capture_output=True, text=True, check=True
        ).stdout
    except subprocess.CalledProcessError as e:
        print("Failed to execute llvm-bolt --help:")
        print(e)
        return

    sections = parse_bolt_options(help_output)
    markdown = generate_markdown(sections)

    with open("CommandLineArgumentReference.md", "w") as md_file:
        md_file.write(markdown)


if __name__ == "__main__":
    main()
