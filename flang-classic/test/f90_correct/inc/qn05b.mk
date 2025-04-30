#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn05b  ########


qn05b: run
	

build:  $(SRC)/qn05b.f90
	-$(RM) qn05b.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn05b.f90 -o qn05b.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn05b.$(OBJX) check.$(OBJX) $(LIBS) -o qn05b.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn05b
	qn05b.$(EXESUFFIX)

verify: ;

qn05b.run: run

