#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn05c  ########


qn05c: run
	

build:  $(SRC)/qn05c.f90
	-$(RM) qn05c.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn05c.f90 -o qn05c.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn05c.$(OBJX) check.$(OBJX) $(LIBS) -o qn05c.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn05c
	qn05c.$(EXESUFFIX)

verify: ;

qn05c.run: run

