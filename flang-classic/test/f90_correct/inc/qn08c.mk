#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn08c  ########


qn08c: run
	

build:  $(SRC)/qn08c.f90
	-$(RM) qn08c.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn08c.f90 -o qn08c.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn08c.$(OBJX) check.$(OBJX) $(LIBS) -o qn08c.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn08c
	qn08c.$(EXESUFFIX)

verify: ;

qn08c.run: run

