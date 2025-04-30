#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn06  ########


qn06: run
	

build:  $(SRC)/qn06.f90
	-$(RM) qn06.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn06.f90 -o qn06.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn06.$(OBJX) check.$(OBJX) $(LIBS) -o qn06.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn06
	qn06.$(EXESUFFIX)

verify: ;

qn06.run: run

