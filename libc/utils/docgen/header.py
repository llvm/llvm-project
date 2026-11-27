# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##
from functools import lru_cache
from pathlib import Path
import re
import sys
from typing import Dict, Generator, List, Optional, Set, Tuple
import yaml

_C_SEC_RE = re.compile(r"^([0-9]+|[FGHJK])\.")


def _stat(statuses: List[str]) -> Dict[str, int]:
    return {
        "total": len(statuses),
        "implemented": sum(s == "implemented" for s in statuses),
        "experimental": sum(s == "experimental" for s in statuses),
    }


def _iter_cmake_entries(
    path: Path, prefix: str
) -> Generator[Tuple[str, bool], None, None]:
    stack: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith(("if(", "if (")):
            stack.append(line)
        elif line.startswith(("elseif(", "else(")) and stack:
            stack[-1] = line
        elif line.startswith("endif(") and stack:
            stack.pop()
        elif line.startswith(prefix):
            yield line[len(prefix) :], any("EXPERIMENTAL" in c for c in stack)


def _is_stub_file(path: Path, is_experimental: bool = False) -> bool:
    text = path.read_text(encoding="utf-8")
    low = text.lower()
    has_marker = (
        any(m in low for m in ("todo", "unimplemented", "for now"))
        or "ENOSYS" in text
        or "[[maybe_unused]]" in text
    )
    if not has_marker and not is_experimental and "math" not in path.parts:
        return False

    idx = text.find("LLVM_LIBC_FUNCTION")
    if idx == -1:
        return False
    open_brace = text.find("{", idx)
    if open_brace == -1:
        return False

    depth, body = 0, ""
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                body = text[open_brace + 1 : i]
                break

    clean = re.sub(r"//.*$", "", body, flags=re.MULTILINE)
    clean = re.sub(r"/\*.*?\*/", "", clean, flags=re.DOTALL).strip()
    stmts = [s.strip() for s in clean.split(";") if s.strip()]
    if len(stmts) == 0:
        return has_marker or is_experimental
    if len(stmts) > 2 or (not has_marker and "math" not in path.parts):
        return False

    for stmt in stmts:
        if stmt.startswith(("libc_errno =", "return", "*signp = 0")):
            no_cast = re.sub(
                r"(const_cast|reinterpret_cast|static_cast)\s*<[^>]+>", "", stmt
            )
            if "::" in no_cast:
                return False
            ret = no_cast.replace("return", "").strip()
            if re.search(r"\b[A-Za-z0-9_]+\s*\(", ret):
                return False
        else:
            return False
    return True


_VARIABLE_RE = re.compile(
    r"LLVM_LIBC_VARIABLE\s*\([^,]+,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)"
)


@lru_cache(maxsize=None)
def _load_implemented_functions(
    libc_root: Path,
) -> Tuple[Set[str], Dict[str, Set[str]], Set[str], Dict[str, Set[str]], Set[str]]:
    raw_def_eps: List[str] = []
    raw_exp_eps: List[str] = []
    def_hdrs: Set[str] = set()
    exp_hdrs: Set[str] = set()
    config_dir = libc_root / "config"
    if config_dir.exists():
        for ep_file in config_dir.rglob("entrypoints.txt"):
            for entry, is_exp in _iter_cmake_entries(ep_file, "libc.src."):
                (raw_exp_eps if is_exp else raw_def_eps).append(entry)
        for hdr_file in config_dir.rglob("headers.txt"):
            for entry, is_exp in _iter_cmake_entries(hdr_file, "libc.include."):
                (exp_hdrs if is_exp else def_hdrs).add(entry)

    exp_targets = {e.split(".")[-1] for e in raw_exp_eps} - {
        e.split(".")[-1] for e in raw_def_eps
    }
    stubs, reals = set(), set()
    vars_by_target: Dict[str, Set[str]] = {}
    src_dir = libc_root / "src"
    if src_dir.exists():
        for p in src_dir.rglob("*.cpp"):
            if _is_stub_file(p, p.stem in exp_targets):
                stubs.add(p.stem)
            else:
                reals.add(p.stem)
                for var in _VARIABLE_RE.findall(p.read_text(encoding="utf-8")):
                    vars_by_target.setdefault(p.stem, set()).add(var)

    stub_only = stubs - reals

    def _collect(entries: List[str]) -> Tuple[Set[str], Dict[str, Set[str]]]:
        syms: Set[str] = set()
        by_stem: Dict[str, Set[str]] = {}
        for entry in entries:
            parts = entry.split(".")
            sym = parts[-1]
            if sym not in stub_only:
                stem = "/".join(parts[:-1])
                exported = {sym} | vars_by_target.get(sym, set())
                syms |= exported
                by_stem.setdefault(stem, set()).update(exported)
        return syms, by_stem

    def_fns, def_by_stem = _collect(raw_def_eps)
    exp_fns, exp_by_stem = _collect(raw_exp_eps)
    return def_fns, def_by_stem, exp_fns - def_fns, exp_by_stem, exp_hdrs - def_hdrs


def _load_include_yaml(yaml_path: Path) -> Dict:
    if not yaml_path.exists():
        return {}
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    for rel in data.get("merge_yaml_files", []):
        merged = _load_include_yaml((yaml_path.parent / rel).resolve())
        for key in ("macros", "types", "enums", "objects", "functions"):
            if key in merged:
                data.setdefault(key, []).extend(merged[key])
    return data


class Header:
    """
    Maintains implementation information about a standard header file:
    * where does its implementation dir live
    * where is its macros file
    * where is its docgen yaml file

    Macro implementation status is checked across header-specific
    ``*-macros.h`` files under ``$LLVM_PROJECT_ROOT/libc/include/llvm-libc-macros``,
    macro and enum entries in ``$LLVM_PROJECT_ROOT/libc/include/<header>.yaml``
    (including shared ``macro_header`` definitions), and ``include/<header>.h.def``
    templates.

    Function implementation status is determined from the union of exported
    entrypoints across ``$LLVM_PROJECT_ROOT/libc/config/**/entrypoints.txt``,
    excluding source files under ``libc/src/`` whose only implementation is a
    ``TODO`` stub.
    """

    def __init__(self, header_name: str, libc_root: Optional[Path] = None):
        """
        :param header_name: e.g., ``"threads.h"`` or ``"signal.h"``
        """
        self.name = header_name
        self.stem = header_name[:-2] if header_name.endswith(".h") else header_name
        self.docgen_root = Path(__file__).parent
        self.libc_root = (
            libc_root if libc_root is not None else self.docgen_root.parent.parent
        )
        self.docgen_yaml = self.docgen_root / Path(header_name).with_suffix(".yaml")
        # Path to libc/include/<name>.yaml
        self.include_yaml = Path(
            self.libc_root, "include", Path(header_name).with_suffix(".yaml")
        )
        self.fns_dir = Path(self.libc_root, "src", self.stem)
        self.macros_dir = Path(self.libc_root, "include", "llvm-libc-macros")

    @property
    def api(self) -> Dict:
        if not hasattr(self, "_api"):
            self._api = (
                yaml.safe_load(self.docgen_yaml.read_text(encoding="utf-8"))
                if self.docgen_yaml.exists()
                else {}
            ) or {}
        return self._api

    @property
    def include_data(self) -> Dict:
        if not hasattr(self, "_include_data"):
            self._include_data = _load_include_yaml(self.include_yaml)
        return self._include_data

    def macro_file_exists(self) -> bool:
        if any(self.__get_macro_files()):
            return True
        return bool(self.include_data.get("macros"))

    def fns_dir_exists(self) -> bool:
        return self.fns_dir.exists() and self.fns_dir.is_dir()

    def is_experimental_header(self) -> bool:
        _, _, _, _, exp_hdrs = _load_implemented_functions(self.libc_root)
        return self.stem.replace("/", "_") in exp_hdrs

    def _in_fn_group(
        self, fn_name: str, all_syms: Set[str], by_stem: Dict[str, Set[str]]
    ) -> bool:
        if fn_name in by_stem.get(self.stem, set()):
            return True
        if not hasattr(self, "_declared_fns"):
            self._declared_fns = {
                f["name"] for f in self.include_data.get("functions", []) if "name" in f
            }
        return fn_name in all_syms and fn_name in self._declared_fns

    def fn_status(self, fn_name: str) -> str:
        def_fns, def_stem, exp_fns, exp_stem, _ = _load_implemented_functions(
            self.libc_root
        )
        if self._in_fn_group(fn_name, def_fns, def_stem):
            return "experimental" if self.is_experimental_header() else "implemented"
        if self._in_fn_group(fn_name, exp_fns, exp_stem):
            return "experimental"
        return "missing"

    def implements_fn(self, fn_name: str) -> bool:
        return self.fn_status(fn_name) == "implemented"

    def _has_macro(self, m_name: str) -> bool:
        macro_re = re.compile(rf"^\s*#\s*define\s+{re.escape(m_name)}\b", re.MULTILINE)
        for f in self.__get_macro_files():
            if macro_re.search(f.read_text(encoding="utf-8")):
                return True

        if any(
            m.get("macro_name") == m_name
            and ("macro_value" in m or "macro_header" in m)
            for m in self.include_data.get("macros", [])
        ):
            return True

        if any(e.get("name") == m_name for e in self.include_data.get("enums", [])):
            return True

        def_file = self.libc_root / "include" / f"{self.name}.def"
        if def_file.exists() and macro_re.search(def_file.read_text(encoding="utf-8")):
            return True

        return False

    def macro_status(self, m_name: str) -> str:
        if not self._has_macro(m_name):
            return "missing"
        return "experimental" if self.is_experimental_header() else "implemented"

    def implements_macro(self, m_name: str) -> bool:
        return self.macro_status(m_name) == "implemented"

    def _load_exposed_types(self) -> Set[str]:
        hdrgen_dir = str(self.docgen_root.parent / "hdrgen")
        if hdrgen_dir not in sys.path:
            sys.path.insert(0, hdrgen_dir)
        from hdrgen.header import COMPILER_HEADER_TYPES, HeaderFile
        from hdrgen.yaml_to_classes import load_yaml_file

        queue = []
        if self.include_yaml.exists() and "header" in self.include_data:
            header_file = load_yaml_file(self.include_yaml, HeaderFile, None)
            queue.extend(t.name for t in header_file.all_types())
        else:
            queue.extend(
                t["type_name"]
                for t in self.include_data.get("types", [])
                if "type_name" in t
            )

        def_file = self.libc_root / "include" / f"{self.name}.def"
        if def_file.exists():
            queue.extend(
                re.findall(
                    r"llvm-libc-types/([A-Za-z0-9_]+)\.h",
                    def_file.read_text(encoding="utf-8"),
                )
            )

        found: Set[str] = set()
        seen: Set[str] = set()
        types_dir = self.libc_root / "include" / "llvm-libc-types"
        while queue:
            type_name = queue.pop()
            if type_name in seen:
                continue
            seen.add(type_name)

            type_path = types_dir / f"{type_name}.h"
            if type_name in COMPILER_HEADER_TYPES or type_path.exists():
                found.add(type_name)
            if type_path.exists():
                queue.extend(
                    re.findall(
                        r'#include\s+"(?:.*/)?([A-Za-z0-9_]+)\.h"',
                        type_path.read_text(encoding="utf-8"),
                    )
                )
        typedef_re = re.compile(
            r"^\s*typedef\b[^;{]*?\b([A-Za-z_][A-Za-z0-9_]*)\s*;", re.MULTILINE
        )
        for mf in self.__get_macro_files():
            found.update(typedef_re.findall(mf.read_text(encoding="utf-8")))
        return found

    def type_status(self, t_name: str) -> str:
        if not hasattr(self, "_exposed_types"):
            self._exposed_types = self._load_exposed_types()
        found = (
            t_name in self._exposed_types
            or t_name.replace(" ", "_") in self._exposed_types
        )
        if not found:
            return "missing"
        return "experimental" if self.is_experimental_header() else "implemented"

    def implements_type(self, t_name: str) -> bool:
        return self.type_status(t_name) == "implemented"

    def object_status(self, o_name: str) -> str:
        if not any(
            o.get("object_name") == o_name for o in self.include_data.get("objects", [])
        ):
            return "missing"
        def_syms, def_stem, exp_syms, exp_stem, _ = _load_implemented_functions(
            self.libc_root
        )
        if o_name in def_stem.get(self.stem, set()) or o_name in def_syms:
            return "experimental" if self.is_experimental_header() else "implemented"
        if o_name in exp_stem.get(self.stem, set()) or o_name in exp_syms:
            return "experimental"
        return "missing"

    def implements_object(self, o_name: str) -> bool:
        return self.object_status(o_name) == "implemented"

    def __get_macro_files(self) -> Generator[Path, None, None]:
        """
        This function uses a glob on, e.g., ``"**/fcntl.macros.h"`` because the
        macro file might be located in a subdirectory:
        libc/include/llvm-libc-macros/fcntl-macros.h
        libc/include/llvm-libc-macros/linux/fcntl-macros.h

        When a header would be nested in a dir (such as arpa/, sys/, etc) we
        instead use a hyphen in the name.
        libc/include/llvm-libc-macros/sys-mman-macros.h
        """
        stem = self.stem.replace("/", "-")
        yield from self.macros_dir.glob(f"**/{stem}-macros.h")
        def_file = self.libc_root / "include" / f"{self.name}.def"
        if def_file.exists():
            for rel in re.findall(
                r'#include\s+"(?:.*/)?llvm-libc-macros/([^"]+)"',
                def_file.read_text(encoding="utf-8"),
            ):
                macro_path = self.macros_dir / rel
                if macro_path.exists():
                    yield macro_path

    def evaluate_family(self, fam_name: str, fam_data: Dict) -> Dict:
        variants = fam_data.get("variants") or {}
        accuracy = fam_data.get("accuracy") or {}
        results = {}
        implemented_count = 0
        exp_count = 0
        for prec, sym in variants.items():
            st = self.fn_status(sym) if sym else "missing"
            if st == "implemented":
                implemented_count += 1
            elif st == "experimental":
                exp_count += 1
            results[prec] = {
                "symbol": sym or "",
                "status": st,
                "implemented": st == "implemented",
                "accuracy": accuracy.get(prec, ""),
            }
        fam_status = (
            "implemented"
            if implemented_count > 0
            else ("experimental" if exp_count > 0 else "missing")
        )
        return {
            "name": fam_name,
            "status": fam_status,
            "implemented": implemented_count > 0,
            "variants": results,
            "implemented_count": implemented_count,
            "total_variants": len(variants),
        }

    def get_summary_stats(self, api: Optional[Dict] = None) -> Dict:
        api = api if api is not None else self.api
        families = api.get("families") or {}
        fam_eval = {
            fn: self.evaluate_family(fn, fd or {}) for fn, fd in families.items()
        }
        precisions = []
        for col in api.get("family-columns") or []:
            vals = [
                r["variants"][col["key"]]["status"]
                for r in fam_eval.values()
                if col["key"] in r["variants"]
            ]
            if vals:
                precisions.append(
                    {"key": col["key"], "label": col["label"], **_stat(vals)}
                )

        all_items: List[Tuple[Dict, str]] = [
            (fd or {}, fam_eval[n]["status"]) for n, fd in families.items()
        ]
        counts = {"families": _stat([s for _, s in all_items])}
        for key, sec_name, status_fn in (
            ("raw_functions", "functions", self.fn_status),
            ("macros", "macros", self.macro_status),
            ("types", "types", self.type_status),
            ("objects", "objects", self.object_status),
        ):
            pairs = [
                (it or {}, status_fn(n)) for n, it in (api.get(sec_name) or {}).items()
            ]
            counts[key] = _stat([s for _, s in pairs])
            all_items.extend(pairs)

        std = api.get("standard", "")
        removed = ("removed-in-posix-2024", "removed-in-posix-2008")
        return {
            "header": self.name,
            "standard": std,
            "posix": api.get("posix", ""),
            "experimental_header": self.is_experimental_header(),
            "removed_posix": not std
            and not api.get("posix")
            and bool(all_items)
            and all(any(k in it for k in removed) for it, _ in all_items),
            "overall": _stat([s for _, s in all_items]),
            "functions": {
                k: counts["families"][k] + counts["raw_functions"][k]
                for k in ("total", "implemented", "experimental")
            },
            **counts,
            "c23": _stat(
                [
                    s
                    for it, s in all_items
                    if std == "c23"
                    and _C_SEC_RE.match(str(it.get("c-definition") or ""))
                ]
            ),
            "tr18037": _stat(
                [
                    s
                    for it, s in all_items
                    if std == "iso-tr-18037"
                    and str(it.get("c-definition") or "").startswith("TR 18037")
                ]
            ),
            "posix_spec": _stat([s for it, s in all_items if "in-latest-posix" in it]),
            "precisions": precisions,
        }
