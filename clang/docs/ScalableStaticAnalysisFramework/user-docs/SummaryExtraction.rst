==================
Summary Extraction
==================

.. WARNING:: The framework is rapidly evolving.
  The documentation might be out-of-sync with the implementation.
  The purpose of this documentation is to give context for upcoming reviews.

Command-line interface
**********************

Two flags control summary extraction:

- ``--ssaf-extract-summaries=<name1>,<name2>,...``: Comma-separated list of summary extractor names to enable.
- ``--ssaf-tu-summary-file=<path>.<format>``: Output file for the extracted summaries. The file extension selects the serialization format (e.g. ``.json``).

Example invocation:

.. code-block:: bash

  clang --ssaf-extract-summaries=MyAwesomeAnalysis \
        --ssaf-tu-summary-file=my-tu-summary.json \
        -c input.cpp -o input.o

Diagnostics
***********

In case the ``--ssaf-*`` flags are used incorrectly, or some extractor fails to implement the desired serialization format
or just happens to have an error, then the error is forwarded as a ``scalable-static-analysis-framework`` error.
These errors can be downgraded into warnings using ``-Wno-error=scalable-static-analysis-framework``.
These errors can be completely suppressed using ``-Wno-scalable-static-analysis-framework``.

See the `diagnostic flags <https://clang.llvm.org/docs/DiagnosticsReference.html#wscalable-static-analysis-framework>`_ for the full list of diagnostics controlled by ``-Wscalable-static-analysis-framework``.
