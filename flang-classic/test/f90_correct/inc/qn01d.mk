#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn01d  ########


qn01d: run
	

build:  $(SRC)/qn01d.f90
	-$(RM) qn01d.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn01d.f90 -o qn01d.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn01d.$(OBJX) check.$(OBJX) $(LIBS) -o qn01d.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn01d
	qn01d.$(EXESUFFIX)

verify: ;

qn01d.run: run

