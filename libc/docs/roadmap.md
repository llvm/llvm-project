(roadmap)=

# Roadmap and Known Issues

This page collects planned features and known problems in LLVM-libc. It is a
working list, and each numbered item is intended to become an issue in the
[issue tracker](https://github.com/llvm/llvm-project/labels/libc). Check the
tracker, or ask the maintainers, before starting on one.

Paths are relative to `libc/` unless they start with `.github/` or `.ci/`.
Counts and line numbers are as of November 2026.

## Planned features

These are missing or incomplete today. They are not non-goals; see
{ref}`design_philosophy` for those.

| Feature | Status |
| --- | --- |
| Dynamic loading | Under development. Never intended to be the primary use case. |
| Time zone database | Planned. The intent is to base the implementation on the one in libc++. |
| Thread cancellation | Low priority. When added, it must not significantly slow down code that does not use it. |
| Message catalogs, Annex K | Low priority. |
| Locales | The `_l` variants of the relevant functions are planned. See {ref}`design_philosophy`. |
| UTF-16 `wchar_t` | In progress. |
| Full build for macOS and Windows | Not designed yet. See {ref}`abi_policy`. |
