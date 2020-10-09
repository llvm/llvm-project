Continuous Integration
======================

Buildbot
--------

LLVM Buildbot is the place where volunteers provide build machines. Everyone can
`add a buildbot for LLDB <https://llvm.org/docs/HowToAddABuilder.html>`_.

* `lldb-x64-windows-ninja <http://lab.llvm.org:8011/#/builders/83>`_
* `lldb-x86_64-debian <http://lab.llvm.org:8011/#/builders/68>`_
* `lldb-aarch64-ubuntu <http://lab.llvm.org:8011/#/builders/96>`_
* `lldb-arm-ubuntu <http://lab.llvm.org:8011/#/builders/17>`_
* `lldb-x86_64-fedora <http://lab.llvm.org:8011/#/builders/22>`_

An overview of all LLDB builders can be found here:

`http://lab.llvm.org:8011/#/builders?tags=lldb <http://lab.llvm.org:8011/#/builders?tags=lldb>`_

GreenDragon
-----------

GreenDragon builds and tests LLDB on macOS. It has a `dedicated tab
<http://green.lab.llvm.org/green/view/LLDB/>`_ for LLDB.

* `lldb-cmake <http://green.lab.llvm.org/green/view/LLDB/job/lldb-cmake/>`_
* `lldb-cmake-matrix <http://green.lab.llvm.org/green/view/LLDB/job/lldb-cmake-matrix/>`_
* `lldb-cmake-reproducers <http://green.lab.llvm.org/green/view/LLDB/job/lldb-cmake-reproducers/>`_
* `lldb-cmake-standalone <http://green.lab.llvm.org/green/view/LLDB/job/lldb-cmake-standalone/>`_
* `lldb-cmake-sanitized <http://green.lab.llvm.org/green/view/LLDB/job/lldb-cmake-sanitized/>`_

Documentation
-------------

The documentation bot validates that the website builds correctly with Sphinx.
It does not generate the website itself, which happens on a separate server.

* `lldb-sphinx-docs <http://lab.llvm.org:8011/builders/lldb-sphinx-docs>`_
