The ground truth of standards
=============================

Like any modern libc, LLVM libc also supports a wide number of standards and
extensions. To avoid developing headers, wrappers and sources in a disjointed
fashion, LLVM libc employs ground truth files. These files live under the
``spec`` directory and list ground truth corresponding the ISO C standard, the
POSIX extension standard, etc. For example, the path to the ground truth file
for the ISO C standard is ``spec/stdc.td``. Tools like the header generator
(described in the header generation document), docs generator, etc. use the
ground truth files to generate headers, docs etc.
