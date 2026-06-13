==============================
Source Edit Generation
==============================

Source edit generation is the second stage of the SSAF pipeline. Given a
``WPASuite`` produced by an earlier whole-program analysis, a *source
transformation* runs alongside the normal compile and emits two
per-translation-unit artifacts:

- a *source-edit file* (``--ssaf-src-edit-file=``) containing
  ``clang::tooling::Replacement`` records ready for
  ``clang-apply-replacements``,
- a *transformation-report file* (``--ssaf-transformation-report-file=``)
  containing diagnostic-style findings.

Driver flags
============

Four flags control the pipeline; they are all both ``--ssaf-…`` driver
flags and ``cc1`` flags. The compilation-unit identifier flag is shared
with the stage-1 pipeline.

.. list-table::
   :header-rows: 1

   * - Flag
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
     - Transformation-report output. Always written as a SARIF 2.1.0
       JSON document; the file extension is not interpreted.
   * - ``--ssaf-compilation-unit-id=<id>``
     - Stable identifier for this translation unit (also required by
       the stage-1 pipeline).

When ``--ssaf-source-transformation=`` is non-empty the framework wraps
the active ``FrontendAction`` in a ``SourceTransformationFrontendAction``;
otherwise the compile is byte-for-byte unchanged.

Error policy
============

Every CLI-misuse and runtime-write diagnostic is registered as a
``Warning ... DefaultError`` under
``-Wscalable-static-analysis-framework``. This means errors stop the
compile by default but can be downgraded or silenced:

- ``-Wno-error=scalable-static-analysis-framework`` — diagnostics
  become warnings and the compile finishes normally. The edit/report
  files may be absent (if the runner bailed out before writing) or
  present (if a write call returned an error).
- ``-Wno-scalable-static-analysis-framework`` — diagnostics are
  silenced entirely. The compile finishes normally.

Examples
========

Apply the source edits with ``clang-apply-replacements``:

.. code-block:: console

   $ clang -c foo.cpp \
       --ssaf-source-transformation=my-transformation \
       --ssaf-global-scope-analysis-result=wpa.json \
       --ssaf-src-edit-file=foo.yaml \
       --ssaf-transformation-report-file=foo.sarif \
       --ssaf-compilation-unit-id=cu-foo
   $ clang-apply-replacements --remove-change-desc-files <dir-with-yaml>

The transformation report can be consumed by any SARIF 2.1.0 viewer.
