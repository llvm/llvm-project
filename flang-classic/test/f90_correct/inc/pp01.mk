#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pp01  ########


pp01: run
	

build:  $(SRC)/pp01.f90
	-$(RM) pp01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pp01.f90 -o pp01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pp01.$(OBJX) check.$(OBJX) $(LIBS) -o pp01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pp01
	pp01.$(EXESUFFIX)

verify: ;

pp01.run: run

