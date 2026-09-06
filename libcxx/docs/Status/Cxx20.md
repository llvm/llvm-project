(cxx20-status)=

# libc++ C++20 Status

:::{include} ../Helpers/Styles.md
:::

:::{contents}
:local: true
:::

## Overview

In July 2017, the C++ standard committee created a draft for the next version of the C++ standard, initially known as "C++2a".
In September 2020, the C++ standard committee approved this draft, and sent it to ISO for approval as C++20.

This page shows the status of libc++; the status of clang's support of the language features is [here](https://clang.llvm.org/cxx_status.html#cxx20).

The groups that have contributed papers:

- CWG - Core Language Working group
- LWG - Library working group
- SG1 - Study group #1 (Concurrency working group)

:::{note}
"Nothing to do" means that no library changes were needed to implement this change.
:::

## Paper Status

```{eval-rst}
.. role:: notstarted
.. role:: nothingtodo
.. role:: inprogress
.. role:: inreview
.. role:: partial
.. role:: complete

.. csv-table::
   :file: Cxx20Papers.csv
   :header-rows: 1
   :widths: auto
```

## Library Working Group Issues Status

```{eval-rst}

.. csv-table::
   :file: Cxx20Issues.csv
   :header-rows: 1
   :widths: auto
```
