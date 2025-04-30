#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn01b  ########


qn01b: run
	

build:  $(SRC)/qn01b.f90
	-$(RM) qn01b.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn01b.f90 -o qn01b.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn01b.$(OBJX) check.$(OBJX) $(LIBS) -o qn01b.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn01b
	qn01b.$(EXESUFFIX)

verify: ;

qn01b.run: run

