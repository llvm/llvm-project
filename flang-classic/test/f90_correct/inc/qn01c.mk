#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn01c  ########


qn01c: run
	

build:  $(SRC)/qn01c.f90
	-$(RM) qn01c.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn01c.f90 -o qn01c.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn01c.$(OBJX) check.$(OBJX) $(LIBS) -o qn01c.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn01c
	qn01c.$(EXESUFFIX)

verify: ;

qn01c.run: run

