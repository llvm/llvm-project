.. title:: clang-tidy - portability-avoid-pragma-once

portability-avoid-pragma-once
=============================

Finds uses of ``#pragma once`` and suggests replacing them with standard
include guards (``#ifndef``/``#define``/``#endif``) for improved portability.

``#pragma once`` is a non-standard extension, despite being widely supported
by modern compilers. Relying on it can lead to portability issues in
some environments.

Some older or specialized C/C++ compilers, particularly in embedded systems,
may not fully support ``#pragma once``.

It can also fail in certain file system configurations, like network drives
or complex symbolic links, potentially leading to compilation issues.

Consider the following header file:

.. code:: c++

  // my_header.h
  #pragma once // warning: avoid 'pragma once' directive; use include guards instead


The warning suggests using include guards:

.. code:: c++

  // my_header.h
  #ifndef PATH_TO_MY_HEADER_H // Good: use include guards.
  #define PATH_TO_MY_HEADER_H

  #endif // PATH_TO_MY_HEADER_H
