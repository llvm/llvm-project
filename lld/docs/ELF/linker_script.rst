Linker Script implementation notes and policy
=============================================

LLD implements a large subset of the GNU ld linker script notation. The LLD
implementation policy is to implement linker script features as they are
documented in the ld `manual <https://sourceware.org/binutils/docs/ld/Scripts.html>`_
We consider it a bug if the lld implementation does not agree with the manual
and it is not mentioned in the exceptions below.

The ld manual is not a complete specification, and is not sufficient to build
an implementation. In particular some features are only defined by the
implementation and have changed over time.

The lld implementation policy for properties of linker scripts that are not
defined by the documentation is to follow the GNU ld implementation wherever
possible. We reserve the right to make different implementation choices where
it is appropriate for LLD. Intentional deviations will be documented in this
file.

Output section description
~~~~~~~~~~~~~~~~~~~~~~~~~~

The description of an output section looks like:

::

  section [address] [(type)] : [AT(lma)] [ALIGN(section_align)] [SUBALIGN](subsection_align)] {
    output-section-command
    ...
  } [>region] [AT>lma_region] [:phdr ...] [=fillexp] [,]

Output section address
----------------------

When an *OutputSection* *S* has ``address``, LLD will set sh_addr to ``address``.

The ELF specification says:

> The value of sh_addr must be congruent to 0, modulo the value of sh_addralign.

The presence of ``address`` can cause the condition unsatisfied. LLD will warn.
GNU ld from Binutils 2.35 onwards will reduce sh_addralign so that
sh_addr=0 (modulo sh_addralign).

Output section alignment
------------------------

sh_addralign of an *OutputSection* *S* is the maximum of
``ALIGN(section_align)`` and the maximum alignment of the input sections in
*S*.

When an *OutputSection* *S* has both ``address`` and ``ALIGN(section_align)``,
GNU ld will set sh_addralign to ``ALIGN(section_align)``.

Output section LMA
------------------

A load address (LMA) can be specified by ``AT(lma)`` or ``AT>lma_region``.

- ``AT(lma)`` specifies the exact load address. If the linker script does not
  have a PHDRS command, then a new loadable segment will be generated.
- ``AT>lma_region`` specifies the LMA region. The lack of ``AT>lma_region``
  means the default region is used. Note, GNU ld propagates the previous LMA
  memory region when ``address`` is not specified. The LMA is set to the
  current location of the memory region aligned to the section alignment.
  If the linker script does not have a PHDRS command, then if
  ``lma_region`` is different from the ``lma_region`` for
  the previous OutputSection a new loadable segment will be generated.

The two keywords cannot be specified at the same time.

If neither ``AT(lma)`` nor ``AT>lma_region`` is specified:

- If the previous section is also in the default LMA region, and the two
  section have the same memory regions, the difference between the LMA and the
  VMA is computed to be the same as the previous difference.
- Otherwise, the LMA is set to the VMA.
