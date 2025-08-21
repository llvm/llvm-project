.. SPDX-FileCopyrightText: 2025 Siemens Corporation and/or its affiliates
.. Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
.. See https://llvm.org/LICENSE.txt for license information.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. title:: clang-tidy - bugprone-method-hiding

bugprone-method-hiding
=========================

Finds derived class methods that hide a (non-virtual) base class method.

In order to be considered "hiding", methods must have the same signature
(i.e. the same name, same number of parameters, same parameter types, etc).
Only checks public, non-templated methods. 