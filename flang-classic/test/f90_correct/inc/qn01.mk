#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn01  ########


qn01: run
	

build:  $(SRC)/qn01.f90
	-$(RM) qn01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn01.f90 -o qn01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn01.$(OBJX) check.$(OBJX) $(LIBS) -o qn01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn01
	qn01.$(EXESUFFIX)

verify: ;

qn01.run: run

