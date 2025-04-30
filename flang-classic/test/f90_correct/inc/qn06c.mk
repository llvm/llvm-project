#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn06c  ########


qn06c: run
	

build:  $(SRC)/qn06c.f90
	-$(RM) qn06c.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn06c.f90 -o qn06c.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn06c.$(OBJX) check.$(OBJX) $(LIBS) -o qn06c.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn06c
	qn06c.$(EXESUFFIX)

verify: ;

qn06c.run: run

