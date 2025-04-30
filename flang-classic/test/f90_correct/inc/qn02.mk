#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn02  ########


qn02: run
	

build:  $(SRC)/qn02.f90
	-$(RM) qn02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn02.f90 -o qn02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn02.$(OBJX) check.$(OBJX) $(LIBS) -o qn02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn02
	qn02.$(EXESUFFIX)

verify: ;

qn02.run: run

