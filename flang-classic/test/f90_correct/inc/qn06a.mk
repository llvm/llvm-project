#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn06a  ########


qn06a: run
	

build:  $(SRC)/qn06a.f90
	-$(RM) qn06a.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn06a.f90 -o qn06a.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn06a.$(OBJX) check.$(OBJX) $(LIBS) -o qn06a.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn06a
	qn06a.$(EXESUFFIX)

verify: ;

qn06a.run: run

