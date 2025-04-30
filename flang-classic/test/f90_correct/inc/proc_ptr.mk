#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test proc_ptr  ########


proc_ptr: run


build:  $(SRC)/proc_ptr.f90
	-$(RM) proc_ptr.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/proc_ptr.f90 -o proc_ptr.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) proc_ptr.$(OBJX) check.$(OBJX) $(LIBS) -o proc_ptr.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test proc_ptr
	proc_ptr.$(EXESUFFIX)

verify: ;

proc_ptr.run: run

