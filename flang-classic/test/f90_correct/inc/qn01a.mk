#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn01a  ########


qn01a: run
	

build:  $(SRC)/qn01a.f90
	-$(RM) qn01a.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn01a.f90 -o qn01a.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn01a.$(OBJX) check.$(OBJX) $(LIBS) -o qn01a.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn01a
	qn01a.$(EXESUFFIX)

verify: ;

qn01a.run: run

