#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn02c  ########


qn02c: run
	

build:  $(SRC)/qn02c.f90
	-$(RM) qn02c.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn02c.f90 -o qn02c.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn02c.$(OBJX) check.$(OBJX) $(LIBS) -o qn02c.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn02c
	qn02c.$(EXESUFFIX)

verify: ;

qn02c.run: run

