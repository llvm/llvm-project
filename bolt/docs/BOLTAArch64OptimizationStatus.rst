=====================================
AArch64 Optimization and Flags Status
=====================================

Overview
--------

This page summarizes default-off BOLT optimization flags that users may
explicitly enable when optimizing AArch64 binaries.

BOLT is to be used with binaries linked with
relocations (``--emit-relocs`` or ``-Wl,-q``) and representative profile data.

Supported Flags
---------------
The following flags are supported for AArch64.

.. list-table::
     :header-rows: 1
     :widths: 34 42
     :align: left

     * - Flag
       - Optimization
     * - | ``--reorder-functions=exec-count|hfsort|cdsort|pettis-hansen|random|user``
         | ``--function-order=<file>``
       - Reorder functions
     * - ``--reorder-blocks=normal|ext-tsp|cache|branch-predictor|reverse|cluster-shuffle``
       - Reorder basic blocks
     * - | ``--split-functions``
         | ``--split-strategy=profile2|random2|randomN|all``
         | ``--split-all-cold``
         | ``--split-eh``
       - Split hot and cold code
     * - | ``--align-blocks``
         | ``--block-alignment=<uint>``
       - Align basic blocks
     * - ``--tail-duplication=aggressive|moderate|cache``
       - Duplicate branch tails
     * - ``--peepholes=double-jumps|tailcall-traps|useless-branches|all``
       - Run peephole optimizations
     * - | ``--inline-all``
         | ``--inline-small-functions``
         | Related options:
         | ``--inline-ap``
         | ``--inline-limit=<uint>``
         | ``--inline-small-functions-bytes=<uint>``
       - Inline functions
     * - ``--icf=safe|all``
       - Fold identical functions

Supported Flags With Limitations
--------------------------------
The following flags are implemented for AArch64, but require specific runtime
or option conditions. Enabling them without the required conditions may report
an error or perform no transformation.

.. list-table::
     :header-rows: 1
     :widths: 30 28 44
     :align: left

     * - Flag
       - Optimization
       - Notes
     * - ``--inline-memcpy``
       - Inline fixed-size ``memcpy`` calls
       - Only applies when the copy size is a known constant; AArch64 skips sizes over 64 bytes.
     * - ``--plt=hot|all``
       - Optimize PLT calls
       - Requires immediate binding. If BOLT cannot update the binary, relink with ``-znow``.
     * - ``--hugify``
       - Place hot code on huge pages
       - Applies to binaries with a recognized entry point; skipped when ``--instrument`` is used.
     * - | ``--reorder-data=<section1,section2,...>``
         | ``--reorder-data-algo=count|funcs``
       - Reorder data sections
       - ``move``, ``split`` and ``aggressive`` disable data reordering.
     * - ``--split-strategy=cdsplit``
       - Split functions using cache-directed splitting
       - Requires ``--compact-code-model`` on AArch64.

Unsupported Flags
-----------------

The following flags are not available for AArch64. ``Not applicable to
AArch64`` means the optimization targets architectural features or mechanisms
that do not apply to AArch64. ``Not implemented for AArch64`` means the
optimization could be relevant, but is not currently implemented for this
target.

.. list-table::
     :header-rows: 1
     :widths: 30 28 42
     :align: left

     * - Flag
       - Optimization
       - Notes
     * - ``--jt-footprint-reduction``
       - Reduce jump-table footprint
       - Not implemented for AArch64.
     * - ``--three-way-branch``
       - Reorder three-way branches
       - Not implemented for AArch64.
     * - ``--simplify-rodata-loads``
       - Replace read-only data loads with constants
       - Not implemented for AArch64.
     * - ``--frame-opt=hot|all``
       - Optimize stack-frame accesses
       - Not implemented for AArch64.
     * - ``--indirect-call-promotion=calls|jump-tables|all``
       - Promote indirect calls
       - Not implemented for AArch64.
     * - ``--memcpy1-spec=<func1,func2:cs1:cs2,...>``
       - Specialize one-byte ``memcpy`` calls
       - Not implemented for AArch64.
     * - ``--reg-reassign``
       - Reassign registers to reduce encoding size
       - Not applicable to AArch64.
     * - ``--cmov-conversion``
       - Convert branches to conditional moves
       - Not applicable to AArch64.
     * - | ``--stoke``
         | ``--stoke-out``
       - Emit STOKE optimization data
       - Not applicable to AArch64.
     * - ``--insert-retpolines``
       - Insert retpolines
       - Not applicable to AArch64.
