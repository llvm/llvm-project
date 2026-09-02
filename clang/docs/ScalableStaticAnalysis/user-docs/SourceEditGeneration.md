# Source Edit Generation

Source edit generation relies on a `WPASuite` result produced by an
earlier whole-program analysis. It runs alongside the normal compile
and emits two per-translation-unit artifacts:

- a *source-edit file* (`--ssaf-src-edit-file=`) containing
  `clang::tooling::Replacement` records ready for
  `clang-apply-replacements`,
- a *transformation-report file* (`--ssaf-transformation-report-file=`)
  containing diagnostic-style findings.

## Driver options

Five options control the pipeline; they are all both `--ssaf-…` driver
options and `cc1` options. The compilation-unit identifier is shared
with the summary extraction step. A given compilation unit needs to
receive the same identifier for both summary extraction and source
edit generation. The link-unit identifier must match the identifier
of the link unit that this compilation unit was linked into.

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - Option
     - Purpose
   * - ``--ssaf-source-transformation=<name>``
     - Name of the transformation to run.
   * - ``--ssaf-global-scope-analysis-result=<path>.<format>``
     - WPASuite input. The extension selects the serialization format.
   * - ``--ssaf-src-edit-file=<path>``
     - Source-edit output. Always written as a
       ``clang-apply-replacements``-compatible YAML document; the
       file extension is not interpreted.
   * - ``--ssaf-transformation-report-file=<path>``
     - Transformation-report output. Always written as a SARIF JSON
       document; the file extension is not interpreted.
   * - ``--ssaf-compilation-unit-id=<id>``
     - Stable identifier for this translation unit (also required by
       the summary extraction).
   * - ``--ssaf-link-unit-id=<id>``
     - Stable identifier of the link unit this translation unit was
       linked into.
```

When `--ssaf-source-transformation=` is non-empty the framework wraps
the active `FrontendAction` in a `SourceTransformationFrontendAction`;
otherwise the compile is byte-for-byte unchanged.

## Examples

Apply the source edits with `clang-apply-replacements`:

```console
$ clang -c foo.cpp \
    --ssaf-source-transformation=my-transformation \
    --ssaf-global-scope-analysis-result=wpa.json \
    --ssaf-src-edit-file=foo.yaml \
    --ssaf-transformation-report-file=foo.sarif \
    --ssaf-compilation-unit-id=cu-foo \
    --ssaf-link-unit-id=lu-foo
$ clang-apply-replacements --remove-change-desc-files <dir-with-yaml>
```

The transformation report can be consumed by any SARIF viewer.

