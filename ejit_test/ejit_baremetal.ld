/*===-- ejit_baremetal.ld - bare-metal registry link test (custom script) -===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions. See https://llvm.org/LICENSE.txt for license information.      *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/*
 * Custom linker script used by the ejit_baremetal_link_test integration test to
 * SIMULATE a bare-metal target that does not rely on the linker auto-defining
 * __start_/__stop_ bound symbols for custom sections.
 *
 * The test compiles its TUs with -enable-ejit-global-ctors=false, so the EJIT
 * runtime cannot use the global-constructor registration path and MUST fall
 * back to walking the static registry tables via
 *     [__start_ejit_bitcode, __stop_ejit_bitcode)
 *     [__start_ejit_period,  __stop_ejit_period)
 * This script provides those four symbols by hand, exactly as a real bare-metal
 * linker script would, instead of depending on the linker's automatic
 * __start_/__stop_ synthesis.
 *
 * "No custom segment": the gathered registry data is parked in a section that
 * is INSERTed right after .rodata, so it rides inside the normal read-only
 * PT_LOAD segment — it does NOT become a separate loadable segment that a
 * minimal loader might not understand. On a real target with a full SECTIONS
 * command you would instead place `KEEP(*(.ejit_bitcode))` / `KEEP(*(.ejit_period))`
 * directly inside your existing .rodata { } block; the INSERT form here is the
 * equivalent that works on top of the host's default linker script.
 *
 * KEEP() stops --gc-sections from dropping the tables (they are only referenced
 * through the linker-defined bound symbols). ALIGN(8) matches ejit_reg_entry_t.
 */

SECTIONS
{
  .ejit_registry : ALIGN(8)
  {
    . = ALIGN(8);
    __start_ejit_bitcode = .;
    KEEP(*(.ejit_bitcode))
    __stop_ejit_bitcode = .;

    . = ALIGN(8);
    __start_ejit_period = .;
    KEEP(*(.ejit_period))
    __stop_ejit_period = .;
    . = ALIGN(8);
  }
} INSERT AFTER .rodata;
