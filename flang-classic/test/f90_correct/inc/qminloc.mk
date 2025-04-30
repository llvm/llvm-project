#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qminloc  ########


qminloc: run
	

build:  $(SRC)/qminloc.f08
	-$(RM) qminloc.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qminloc.f08 -o qminloc.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qminloc.$(OBJX) check.$(OBJX) $(LIBS) -o qminloc.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test qminloc 
	qminloc.$(EXESUFFIX)

verify: ;


