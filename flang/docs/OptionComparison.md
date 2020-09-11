<!--===- docs/OptionComparison.md 
  
   Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
   See https://llvm.org/LICENSE.txt for license information.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  
-->

# Compiler options comparison

```eval_rst
.. contents::
   :local:
```

This document catalogs the options processed by F18's peers/competitors.  Much of the document is taken up by a set of tables that list the options categorized into different topics.  Some of the table headings link to more information about the contents of the tables.  For example, the table on **Standards conformance** options links to [notes on Standards conformance](#standards).

**There's also important information in the ___[Appendix section](#appendix)___ near the end of the document on how this data was gathered and what ___is___ and ___is not___ included in this document.**  

Note that compilers may support language features without having an option for them.  Such cases are frequently, but not always noted in this document.

## Categorisation of Options

<table>
  <tr>
   <td colspan="7" ><strong><a href="#standards">Standards conformance</a></strong>
   </td>
  </tr>
  <tr>
   <td><strong>Option</strong> </td>
   <td><strong>Cray</strong> </td>
   <td><strong>GNU</strong> </td>
   <td><strong>IBM</strong> </td>
   <td><strong>Intel</strong> </td>
   <td><strong>PGI</strong> </td>
   <td><strong>Flang</strong> </td>
  </tr>
  <tr>
   <td>Overall conformance </td>
   <td>en,
<p>
eN
   </td>
   <td>std=<em>level</em> </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=210">qlanglvl</a>, <a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=257">qsaa</a>
   </td>
   <td><a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-stand">stand level</a>
   </td>
   <td>Mstandard
   </td>
   <td>Mstandard
   </td>
  </tr>
  <tr>
   <td>Compatibility with previous standards or implementations
   </td>
   <td>N/A
   </td>
   <td>fdec,
<p>
fall-instrinsics
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=297">qxlf77</a>,
<p>
<a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=299">qxlf90</a>,
<p>
<a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=301">qxlf2003</a>,
<p>
<a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=305">qxfl2008</a>,
<p>
<a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=245">qport</a>
   </td>
   <td><a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-f66">f66</a>,
<p>
<a href="https://software.intel.com/node/a8bfa478-22d5-4000-b0ac-b881804a7611">f77rtl</a>,
<p>
<a href="https://software.intel.com/node/647820a0-fc53-4252-a858-46feb012a281">fpscomp</a>,
<p>
<a href="https://software.intel.com/node/e610682a-00fe-4881-9cf7-8eee08c5f2a2">Intconstant</a>,
<p>
<a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-standard-realloc-lhs">nostandard-realloc-lhs</a>,
<p>
<a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-standard-semantics">standard-semantics</a>,
<p>
<a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-assume">assume nostd_intent_in</a>,
<p>
<a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-assume">assume nostd_value</a>,
<p>
<a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-assume">assume norealloc_lhs</a>
   </td>
   <td>Mallocatable=95|03
   </td>
   <td>Mallocatable=95|03
   </td>
  </tr>
</table>





<table>
  <tr>
   <td colspan="7" ><strong><a href="#source">Source format</a></strong>
   </td>
  </tr>
  <tr>
   <td><strong>Option</strong>
   </td>
   <td><strong>Cray</strong>
   </td>
   <td><strong>GNU</strong>
   </td>
   <td><strong>IBM</strong>
   </td>
   <td><strong>Intel</strong>
   </td>
   <td><strong>PGI</strong>
   </td>
   <td><strong>Flang</strong>
   </td>
  </tr>
  <tr>
   <td>Fixed or free source
   </td>
   <td>f free,
<p>
f fixed
   </td>
   <td>ffree-form,
<p>
ffixed-form
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=179">qfree</a>,
<p>
<a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=169">qfixed</a>
   </td>
   <td><a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-fixed">fixed</a>,
<p>
<a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-free">free</a>
   </td>
   <td>Mfree,
<p>
Mfixed
   </td>
   <td>Mfreeform,
<p>
Mfixed
   </td>
  </tr>
  <tr>
   <td>Source line length
   </td>
   <td>N <em>col</em>
   </td>
   <td>ffixed-line-length-n,
<p>
ffree-line-length-n
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=169">qfixed=n</a>
   </td>
   <td><a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-extend-source">extend-source [size]</a>
   </td>
   <td>Mextend
   </td>
   <td>Mextend
   </td>
  </tr>
  <tr>
   <td>Column 1 comment specifier
   </td>
   <td>ed
   </td>
   <td>fd-lines-as-code,
<p>
fd-lines-as-comments
   </td>
   <td>D,
<p>
<a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=161">qdlines</a>,
<p>
<a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=306">qxlines</a>
   </td>
   <td><a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-d-lines-qd-lines">d-lines</a>
   </td>
   <td>Mdlines
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Don't treat CR character as a line terminator
   </td>
   <td>NA
   </td>
   <td>N/A
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=153">qnocr</a>
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Source file naming
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=282">qsuffix</a>
   </td>
   <td><a href="https://software.intel.com/node/84174680-3c7d-4225-9611-6083f496aa9b">extfor</a>,
<p>
<a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-tf">Tf filename</a>
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
</table>





<table>
  <tr>
   <td colspan="7" ><strong><a href="#names">Names, Literals, and other tokens</a></strong>
   </td>
  </tr>
  <tr>
   <td><strong>Option</strong>
   </td>
   <td><strong>Cray</strong>
   </td>
   <td><strong>GNU</strong>
   </td>
   <td><strong>IBM</strong>
   </td>
   <td><strong>Intel</strong>
   </td>
   <td><strong>PGI</strong>
   </td>
   <td><strong>Flang</strong>
   </td>
  </tr>
  <tr>
   <td>Max identifier length
   </td>
   <td>N/A
   </td>
   <td>fmax-identifier-length=<em>n</em>
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>"$" in symbol names
   </td>
   <td>N/A
   </td>
   <td>fdollar-ok
   </td>
   <td>default
   </td>
   <td>default
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Allow names with leading "_"
   </td>
   <td>eQ
   </td>
   <td>N/A 
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Specify name format
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=296">U</a>
   </td>
   <td><a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-names">names=keyword</a>
   </td>
   <td>Mupcase
   </td>
   <td>NA
   </td>
  </tr>
  <tr>
   <td>Escapes in literals
   </td>
   <td>N/A
   </td>
   <td>fbackslash
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=163">qescape</a>
   </td>
   <td><a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-assume">assume bscc</a>
   </td>
   <td>Mbackslash
   </td>
   <td>Mbackslash
   </td>
  </tr>
  <tr>
   <td>Allow multibyte characters in strings
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=223">qmbcs</a>
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Create null terminated strings
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=228">qnullterm</a>
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Character to use for "$"
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>Mdollar,<em>char</em>
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Allow PARAMETER statements without parentheses
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td><a href="https://software.intel.com/node/42e8ec08-64cc-44ba-a636-20ed50c682cd">altparam</a>
   </td>
   <td>N?A
   </td>
   <td>N/A
   </td>
  </tr>
</table>



<table>
  <tr>
   <td colspan="7" ><strong><a href="#do">DO loop handling</a></strong>
   </td>
  </tr>
  <tr>
   <td><strong>Option</strong>
   </td>
   <td><strong>Cray</strong>
   </td>
   <td><strong>GNU</strong>
   </td>
   <td><strong>IBM</strong>
   </td>
   <td><strong>Intel</strong>
   </td>
   <td><strong>PGI</strong>
   </td>
   <td><strong>Flang</strong>
   </td>
  </tr>
  <tr>
   <td>One trip DO loops
   </td>
   <td>ej
   </td>
   <td>N/A
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=113">1</a>,
<p>
<a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=231">qonetrip</a>
   </td>
   <td><a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-f66">f66</a>
   </td>
   <td>Monetrip
   </td>
   <td>N/A 
   </td>
  </tr>
  <tr>
   <td>Allow branching into loops
   </td>
   <td>eg
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
</table>





<table>
  <tr>
   <td colspan="7" ><strong><a href="#real">REAL, DOUBLE PRECISION, and COMPLEX Data</a></strong>
   </td>
  </tr>
  <tr>
   <td><strong>Option</strong>
   </td>
   <td><strong>Cray</strong>
   </td>
   <td><strong>GNU</strong>
   </td>
   <td><strong>IBM</strong>
   </td>
   <td><strong>Intel</strong>
   </td>
   <td><strong>PGI</strong>
   </td>
   <td><strong>Flang</strong>
   </td>
  </tr>
  <tr>
   <td>Default REAL size
   </td>
   <td>s real32,
<p>
s real64,
<p>
s default32,
<p>
s default64
   </td>
   <td>fdefault-real-[8|10|16]
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=252">qrealsize=[4|8]</a>
   </td>
   <td><a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-real-size">real-size [32|64|128]</a>
   </td>
   <td>r[4|8]
   </td>
   <td>r8,
<p>
fdefault-real-8
   </td>
  </tr>
  <tr>
   <td>Default DOUBLE PRECISION size
   </td>
   <td>ep
   </td>
   <td>fdefault-double-8
   </td>
   <td>N/A
   </td>
   <td><a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-double-size">double-size[64|128]</a>
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Make real constants DOUBLE PRECISION
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=161">qdpc</a>
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Promote or demote REAL type sizes
   </td>
   <td>N/A
   </td>
   <td>freal-[4|8]-real[4|8|10|16]
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=144">qautodbl=size</a>
   </td>
   <td>N/A
   </td>
   <td>Mr8,
<p>
Mr8intrinsics
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Rounding mode
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=190">qieee</a>
   </td>
   <td><a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-assume">assume std_minus0_rounding</a>
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Treatment of -0.0
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td><a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-assume">assume minus0</a>
   </td>
   <td>
   </td>
   <td>
   </td>
  </tr>
</table>





<table>
  <tr>
   <td colspan="7" ><strong><a href="#integer">INTEGER and LOGICAL Data</a></strong>
   </td>
  </tr>
  <tr>
   <td><strong>Option</strong>
   </td>
   <td><strong>Cray</strong>
   </td>
   <td><strong>GNU</strong>
   </td>
   <td><strong>IBM</strong>
   </td>
   <td><strong>Intel</strong>
   </td>
   <td><strong>PGI</strong>
   </td>
   <td><strong>Flang</strong>
   </td>
  </tr>
  <tr>
   <td>Default INTEGER size
   </td>
   <td>s integer32,
<p>
s integer64,
<p>
s default32,
<p>
s default64
   </td>
   <td>fdefault-integer-8
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=202">qintsize=[2|4|8]</a>
   </td>
   <td><a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-integer-size">integer-size [32|64|128]</a>
   </td>
   <td>I[2|4|8],
<p>
Mi4,
<p>
Mnoi4
   </td>
   <td>i8,
<p>
fdefault-integer-8
   </td>
  </tr>
  <tr>
   <td>Promote INTEGER sizes
   </td>
   <td>N/A
   </td>
   <td>finteger-4-integer-8
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Enable 8 and 16 bit INTEGER and LOGICALS
   </td>
   <td>eh
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Change how the compiler treats LOGICAL
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>Munixlogical
   </td>
   <td>
   </td>
  </tr>
  <tr>
   <td>Treatment of numeric constants as arguments
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=297">qxlf77 oldboz</a>
   </td>
   <td><a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-assume">assume old_boz</a>
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Treatment of assignment between numerics and logicals
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td><a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-assume">assume old_logical_assign</a>
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
</table>





<table>
  <tr>
   <td colspan="7" ><strong>CHARACTER and Pointer Data</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Option</strong>
   </td>
   <td><strong>Cray</strong>
   </td>
   <td><strong>GNU</strong>
   </td>
   <td><strong>IBM</strong>
   </td>
   <td><strong>Intel</strong>
   </td>
   <td><strong>PGI</strong>
   </td>
   <td><strong>Flang</strong>
   </td>
  </tr>
  <tr>
   <td>Use bytes for pointer arithmetic
   </td>
   <td>s byte_pointer
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Use words for pointer arithmetic
   </td>
   <td>S word_pointer
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Allow character constants for typeless constants
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=153">qctyplss</a>
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
</table>

<table>
  <tr>
   <td colspan="7" ><strong>Data types and allocation</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Option</strong>
   </td>
   <td><strong>Cray</strong>
   </td>
   <td><strong>GNU</strong>
   </td>
   <td><strong>IBM</strong>
   </td>
   <td><strong>Intel</strong>
   </td>
   <td><strong>PGI</strong>
   </td>
   <td><strong>Flang</strong>
   </td>
  </tr>
  <tr>
   <td>Default to IMPLICIT NONE
   </td>
   <td>eI
   </td>
   <td>fimplicit-none
   </td>
   <td>u, <a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=291">qundef</a>
   </td>
   <td><a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-warn">warn declarations</a>
   </td>
   <td>Mdclchk
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Enable DEC STRUCTURE extensions
   </td>
   <td>N/A
   </td>
   <td><a href="https://gcc.gnu.org/onlinedocs/gfortran/STRUCTURE-and-RECORD.html">fdec-structure</a>
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td><a href="https://www.pgroup.com/doc/pgi15fortref.pdf#page=86">default</a>
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Enable <a href="https://pubs.cray.com/content/S-3901/8.7/cray-fortran-reference-manual/types">Cray pointers</a>
   </td>
   <td>default
   </td>
   <td>fcray-pointer
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024776&aid=1#page=432">Default (near equivalent)</a>
   </td>
   <td><a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-pointer-integer">Default (near equivalent)</a>
   </td>
   <td>Mcray
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Allow bitwise logical operations on numeric
   </td>
   <td>ee
   </td>
   <td>N/A
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=201">qintlog</a>
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Allow DEC STATIC and AUTOMATIC declarations
   </td>
   <td>default
   </td>
   <td>fdec-static
   </td>
   <td>Default, see <a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024776&aid=1#page=393">IMPLICIT STATIC</a> and <a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024776&aid=1#page=393">IMPLICIT AUTOMATIC</a>
   </td>
   <td>Default, see <a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-automatic">AUTOMATIC</a> and <a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-static-1#408BC4E6-7AA7-4475-A280-205D34AE2E4F">STATIC</a>
   </td>
   <td><a href="https://www.pgroup.com/doc/pgi15fortref.pdf#page=86">Default</a>
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Allocate variables to static storage
   </td>
   <td>ev
   </td>
   <td>fno-automatic
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=258">qsave</a>
   </td>
   <td><a href="https://software.intel.com/node/3ed16417-6eed-4e09-9edd-4ae03e77c6cf">save</a>,
<p>
<a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-auto">noauto</a>
   </td>
   <td>Mnorecursive,
<p>
Msave
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Compile procedures as if RECURSIVE
   </td>
   <td>eR
   </td>
   <td>frecursive
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=254">q recur</a>
   </td>
   <td><a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-assume">assume recursion</a>,
<p>
<a href="https://software.intel.com/node/1c3820b4-42a1-48ea-887a-2cf39a41ce53">recursive</a>
   </td>
   <td>Mrecursive
   </td>
   <td>Mrecursive
   </td>
  </tr>
</table>


<table>
  <tr>
   <td colspan="7" ><strong><a href="#arrays">Arrays</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Option</strong>
   </td>
   <td><strong>Cray</strong>
   </td>
   <td><strong>GNU</strong>
   </td>
   <td><strong>IBM</strong>
   </td>
   <td><strong>Intel</strong>
   </td>
   <td><strong>PGI</strong>
   </td>
   <td><strong>Flang</strong>
   </td>
  </tr>
  <tr>
  <tr>
   <td>Enable coarrays
   </td>
   <td>h caf
   </td>
   <td>fcoarray=<em>key</em>
   </td>
   <td>N/A
   </td>
   <td><a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-coarray-qcoarray">coarray[=keyword]</a>
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Contiguous array pointers
   </td>
   <td>h contiguous
   </td>
   <td>N/A
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=142">qassert=contig</a>
   </td>
   <td><a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-assume">assume contiguous_pointer</a>
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Contiguous assumed shape dummy arguments
   </td>
   <td>h contiguous_assumed_shape
   </td>
   <td>frepack-arrays
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=142">qassert=contig</a>
   </td>
   <td><a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-assume">assume contiguous_assumed_shape</a>
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
</table>




<table>
  <tr>
   <td colspan="7" ><strong>OpenACC, OpenMP, and CUDA</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Option</strong>
   </td>
   <td><strong>Cray</strong>
   </td>
   <td><strong>GNU</strong>
   </td>
   <td><strong>IBM</strong>
   </td>
   <td><strong>Intel</strong>
   </td>
   <td><strong>PGI</strong>
   </td>
   <td><strong>Flang</strong>
   </td>
  </tr>
  <tr>
   <td>Enable OpenACC
   </td>
   <td> <a href="https://pubs.cray.com/content/S-3901/8.7/cray-fortran-reference-manual/program-model-specific-options">h acc</a>
   </td>
   <td>fopenacc
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td><a href="https://www.pgroup.com/resources/docs/19.1/x86/pgi-ref-guide/index.htm#acc">acc</a>
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Enable OpenMP
   </td>
   <td> <a href="https://pubs.cray.com/content/S-3901/8.7/cray-fortran-reference-manual/program-model-specific-options">h omp</a>
   </td>
   <td>fopenmp
   </td>
   <td><a href="https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=285">qswapomp</a>
   </td>
   <td><a href="https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-qopenmp-qopenmp">qopenmp</a>,
<p>
<a href="https://software.intel.com/node/26d2aef3-9c68-4ef2-9241-bce168a48629">qopenmp-lib</a>,
<p>
<a href="https://software.intel.com/node/f8d22e40-7ed0-4ec6-8e80-d4d27010ca8b">qopenmp-link</a>,
<p>
<a href="https://software.intel.com/node/2cc99a64-9605-44e2-b40a-1409ba459a62">qopenmp-offload</a>,
<p>
<a href="https://software.intel.com/node/02ef8658-3e66-4634-8a11-68eec60322c1">qopenmp-simd</a>,
<p>
<a href="https://software.intel.com/node/088272f2-5eae-42c2-a053-20f42022343d">qopenmp-stubs</a>,
<p>
<a href="https://software.intel.com/node/0ed8bee0-62b4-4266-8cbb-f25b585d4800">qopenmp-threadprivate</a>
   </td>
   <td><a href="https://www.pgroup.com/resources/docs/19.1/x86/pgi-ref-guide/index.htm#mp">mp</a>,
<p>
Mcuda
   </td>
   <td>-mp
   </td>
  </tr>
</table>



<table>
  <tr>
   <td colspan="7" ><strong><a href="#miscellaneous">Miscellaneous</a></strong>
   </td>
  </tr>
  <tr>
   <td><strong>Option</strong>
   </td>
   <td><strong>Cray</strong>
   </td>
   <td><strong>GNU</strong>
   </td>
   <td><strong>IBM</strong>
   </td>
   <td><strong>Intel</strong>
   </td>
   <td><strong>PGI</strong>
   </td>
   <td><strong>Flang</strong>
   </td>
  </tr>
  <tr>
   <td>Disable compile time range checking
   </td>
   <td> N/A
   </td>
   <td>fno-range-check
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Disable call site checking
   </td>
   <td>dC
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Warn for bad call checking
   </td>
   <td>eb
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Set default accessibility of module entities to PRIVATE
   </td>
   <td>N/A
   </td>
   <td>fmodule-private
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
  <tr>
   <td>Force FORALL to use temp
   </td>
   <td>N/A
   </td>
   <td>ftest-forall-temp
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
   <td>N/A
   </td>
  </tr>
</table>



## Notes

**<a name="standards"></a>Standards conformance:** 

All conformance options are similar -- they issue warnings if non-standard features are used.  All defaults are to allow extensions without warnings.  The GNU, IBM, and Intel compilers allow multiple standard levels to be specified.



*   **Cray**: The capital "-eN" option specifies to issue error messages for non-compliance rather than warnings.  
*   **GNU:** The "std=_level_" option specifies the standard to which the program is expected to conform.   The default value for std is 'gnu', which specifies a superset of the latest Fortran standard that includes all of the extensions supported by GNU Fortran, although warnings will be given for obsolete extensions not recommended for use in new code. The 'legacy' value is equivalent but without the warnings for obsolete extensions. The 'f95', 'f2003', 'f2008', and 'f2018' values specify strict conformance to the respective standards.  Errors are given for all extensions beyond the relevant language standard, and warnings are given for the Fortran 77 features that are permitted but obsolescent in later standards. '-std=f2008ts' allows the Fortran 2008 standard including the additions of the Technical Specification (TS) 29113 on Further Interoperability of Fortran with C and TS 18508 on Additional Parallel Features in Fortran.  Values for "_level_" are f_95, f2003, f2008, f2008ts, f2018, gnu,_ and _legacy._

**<a name="source"></a>Source format:** 

**Fixed or free source:**  Cray, IBM, and Intel default the source format based on the source file suffix as follows:



*   **Cray**
    *   **Free:** .f90, .F90, .f95, .F95, .f03, .F03, .f08, .F08, .ftn, .FTN
    *   **Fixed:** .f, .F, .for, .FOR
*   **Intel**
    *   **Free:** .f90, .F90, .i90
    *   **Fixed:** .f, .for, .FOR, .ftn, .FTN, .fpp, .FPP, .i

IBM Fortran's options allow the source line length to be specified with the option, e.g., "-qfixed=72".  IBM bases the default on the name of the command used to invoke the compiler.  IBM has 16 different commands that invoke the Fortran compiler, and the default use of free or fixed format and the line length are based on the command name.  -qfixed=72 is the default for the xlf, xlf_r, f77, and fort77 commands. -qfree=f90is the default for the f90, xlf90, xlf90_r, f95, xlf95, xlf95_r, f2003, xlf2003, xlf2003_r, f2008, xlf2008, and xlf2008_r commands.  The maximum line length for either source format is 132 characters.

**Column 1 comment specifier:**  All compilers allow "D" in column 1 to specify that the line contains a comment and have this as the default for fixed format source.  IBM also supports an "X" in column 1 with the option "-qxlines".

**Source line length:**


*   **Cray:** The "-N _col_" option specifies the line width for fixed- and free-format source lines. The value used for col specifies the maximum number of columns per line.  For free form sources, col can be set to 132, 255, or 1023.  For fixed form sources, col can be set to 72, 80, 132, 255, or 1023.  Characters in columns beyond the col specification are ignored.  By default, lines are 72 characters wide for fixed-format sources and 255 characters wide for free-form sources.
*   **GNU:** For both "ffixed-line-length-_n_" and "ffree-line-length-_n_" options, characters are ignored after the specified length.  The default for fixed is 72.  The default for free is 132.  For free, you can specify 'none' as the length, which means that all characters in the line are meaningful.
*   **IBM:** For **fixed**, the default is 72.  For **free**, there's no default, but the maximum length for either form is 132.
*   **Intel:** The default is 72 for **fixed** and 132 for **free**.
*   **PGI, Flang:** 
    * in free form, it is an error if the line is longer than 1000 characters
    * in fixed form by default, characters after column 72 are ignored
    * in fixed form with -Mextend, characters after column 132 are ignored

**<a name="names"></a>Names, Literals, and other tokens**

**Escapes in literals:**


*   **GNU:** The "-fbackslash" option the interpretation of backslashes in string literals from a single backslash character to "C-style" escape characters. The following combinations are expanded \a, \b, \f, \n, \r, \t, \v, \\, and \0 to the ASCII characters alert, backspace, form feed, newline, carriage return, horizontal tab, vertical tab, backslash, and NUL, respectively. Additionally, \xnn, \unnnn and \Unnnnnnnn (where each n is a hexadecimal digit) are translated into the Unicode characters corresponding to the specified code points. All other combinations of a character preceded by \ are unexpanded.
*   **Intel:** The option "-assume bscc" tells the compiler to treat the backslash character (\) as a C-style control (escape) character syntax in character literals. "nobscc" specifies that the backslash character is treated as a normal character in character literals.  This is the default.

**"$" in symbol names:** Allowing "$" in names is controlled by an option in GNU and is the default behavior in IBM and Intel.  Presumably, these compilers issue warnings when standard conformance options are enabled.  Dollar signs in names don't seem to be allowed in Cray, PGI, or Flang.

**<a name="do"></a>DO loop handling**

**One trip:**



*   **IBM:** IBM has two options that do the same thing: "-1" and "-qonetrip".
*   **Intel:** Intel used to support a "-onetrip" option, but it has been removed.  Intel now supports a "-f66" option that ensures that DO loops are executed at least once in addition to [several other Fortran 66 semantic features](https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-f66#320D769C-7C41-4A84-AE0E-50A72296A838).

**<a name="real"></a>REAL, DOUBLE PRECISION, and COMPLEX Data**

These size options affect the sizes of variables, literals, and intrinsic function results.

**Default REAL sizes:** These options do not affect the size of explicitly declared data (for example, REAL(KIND=4).



*   **Cray:** The "-s default32" and "-s default64" options affect both REAL, INTEGER, and LOGICAL types.

**Default DOUBLE PRECISION:** These options allow control of the size of DOUBLE PRECISION types in conjunction with controlling REAL types.



*   **Cray:** The "-ep" option controls DOUBLE PRECISION. This option can only be enabled when the default data size is 64 bits ("-s default64" or "-s real64").  When "-s default64" or "-s real64" is specified, and double precision arithmetic is disabled, DOUBLE PRECISION variables and constants specified with the D exponent are converted to default real type (64-bit). If double precision is enabled ("-ep"), they are handled as a double precision type (128-bit).  Similarly when the "-s default64" or" -s real64" option is used, variables declared on a DOUBLE COMPLEX statement and complex constants specified with the D exponent are mapped to the complex type in which each part has a default real type, so the complex variable is 128-bit. If double precision is enabled ("-ep"), each part has double precision type, so the double complex variable is 256-bit.
*   **GNU:** The "-fdefault-double-8" option sets the DOUBLE PRECISION type to an 8 byte wide type. Do nothing if this is already the default. If "-fdefault-real-8" is given, DOUBLE PRECISION would instead be promoted to 16 bytes if possible, and "-fdefault-double-8" can be used to prevent this. The kind of real constants like 1.d0 will not be changed by "-fdefault-real-8" though, so also "-fdefault-double-8" does not affect it.

**Promote or demote REAL type sizes:** These options change the meaning of data types specified by declarations of the form REAL(KIND=_N_), except, perhaps for PGI.

*   **GNU:** The allowable combinations are "-freal-4-real-8", "-freal-4-real-10", "-freal-4-real-16", "-freal-8-real-4", "-freal-8-real-10", and "-freal-8-real-16".
*   **IBM:** The "-qautodbl" option  is documented [here](https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=144).
*   **PGI:** The "-Mr8" option promotes REAL variables and constants to DOUBLE PRECISION variables and constants, respectively. DOUBLE PRECISION elements are 8 bytes in length.  The "-Mr8intrinsics" option promotes the intrinsics CMPLX and REAL as DCMPLX and DBLE, respectively.

**<a name="integer"></a>INTEGER and LOGICAL Data**

These size options affect the sizes of variables, literals, and intrinsic function results.

**Default INTEGER sizes:** For all compilers, these options affect both INTEGER and LOGICAL types.

**Enable 8 and 16 bit INTEGER and LOGICAL:** This Cray option ("-eh") enables support for 8-bit and 16-bit INTEGER and LOGICAL types that use explicit kind or star values.  By default ("-eh"), data objects declared as INTEGER(kind=1) or LOGICAL(kind=1) are 8 bits long, and objects declared as INTEGER(kind=2) or LOGICAL(kind=2) are 16 bits long. When this option is disabled ("-dh"), data objects declared as INTEGER(kind=1), INTEGER(kind=2), LOGICAL(kind=1), or LOGICAL(kind=2) are 32 bits long.

**Intrinsic functions**

GNU is the only compiler with options governing the use of non-standard intrinsics.  For more information on the GNU options, see [here](https://gcc.gnu.org/onlinedocs/gcc-8.3.0/gfortran/Fortran-Dialect-Options.html#Fortran-Dialect-Options).  All compilers implement non-standard intrinsics but don't have options that affect access to them.

**<a name="arrays"></a>Arrays**

**Contiguous array pointers:** All vendors that implement this option (Cray, IBM, and Intel) seem to have apply to all pointer targets.  Assuming that the arrays that are targeted by the pointers allows greater optimization.

**Contiguous assumed shape dummy arguments:** Cray and Intel have a separate argument that's specific to assumed shape dummy arguments.

**<a name="miscellaneous"></a>Miscellaneous**

**Disable call site checking:**  This Cray option ("-dC") disables some types of standard call site checking. The current Fortran standard requires that the number and types of arguments must agree between the caller and callee. These constraints are enforced in cases where the compiler can detect them, however, specifying "-dC" disables some of this error checking, which may be necessary in order to get some older Fortran codes to compile.  If error checking is disabled, unexpected compile-time or run time results may occur.  The compiler by default attempts to detect situations in which an interface block should be specified but is not. Specifying "-dC" disables this type of checking as well.

**Warn for bad call checking**: This Cray option ("-eb") issues a warning message rather than an error message when the compiler detects a call to a procedure with one or more dummy arguments having the TARGET, VOLATILE or ASYNCHRONOUS attribute and there is not an explicit interface definition.


## Appendix


### What is and is not included

This document focuses on options relevant to the Fortran language.  This includes some features (such as recursion) that are only indirectly related.  Options related to the following areas are not included:



*   Input/Output
*   Optimization
*   Preprocessing
*   Inlining
*   Alternate library definition or linking
*   Choosing file locations for compiler input or output
*   Modules
*   Warning and error messages and listing output
*   Data initialization
*   Run time checks
*   Debugging
*   Specification of operating system
*   Target architecture
*   Assembler generation
*   Threads or parallelization
*   Profiling and code coverage


### Data sources

Here's the list of compilers surveyed, hot linked to the source of data on it.  Note that this is the only mention of the Oracle and NAG compilers in this document.



*   [Cray Fortran Reference Manual version 8.7](https://pubs.cray.com/content/S-3901/8.7/cray-fortran-reference-manual/compiler-command-line-options)
*   IBM  (XLF) version 14.1 -- [Compiler Referenc](https://www-01.ibm.com/support/docview.wss?uid=swg27024803&aid=1#page=93)e, [Language Reference](https://www-01.ibm.com/support/docview.wss?uid=swg27024776&aid=1)
*   [Intel Fortran version 19.0](https://software.intel.com/en-us/fortran-compiler-developer-guide-and-reference-alphabetical-list-of-compiler-options)
*   [GNU Fortran Compiler version 8.3.0](https://gcc.gnu.org/onlinedocs/gcc-8.3.0/gfortran/Option-Summary.html)
*   [NAG Fortran Release 6.2](https://www.nag.co.uk/nagware/np/r62_doc/manual/compiler_2_4.html)
*   [Oracle Fortran version 819-0492-10](https://docs.oracle.com/cd/E19059-01/stud.10/819-0492/3_options.html)
*   PGI -- [Compiler Reference version 19.1](https://www.pgroup.com/resources/docs/19.1/x86/pgi-ref-guide/index.htm#cmdln-options-ref), [Fortran Reference Guide version 17](https://www.pgroup.com/doc/pgi17fortref.pdf)
*   [Flang](https://github.com/flang-compiler/flang/wiki/Using-Flang) -- information from GitHub

This document has been kept relatively small by providing links to much of the information about options rather than duplicating that information.  For IBM, Intel, and some PGI options, there are direct links.  But direct links were not possible for Cray, GNU and some PGI options.

Many compilers have options that can either be enabled or disabled.  Some compilers indicate this by the presence or absence of the letters "no" in the option name (IBM, Intel, and PGI) while Cray precedes many options with either "e" for enabled or "d" for disabled.  This document only includes the enabled version of the option specification.

Deprecated options were generally ignored, even though they were documented.
