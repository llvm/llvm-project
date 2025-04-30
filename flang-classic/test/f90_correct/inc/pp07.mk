#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pp07  ########


pp07: run
	

build:  $(SRC)/pp07.f90
	-$(RM) pp07.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pp07.f90 -o pp07.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pp07.$(OBJX) check.$(OBJX) $(LIBS) -o pp07.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pp07
	pp07.$(EXESUFFIX)

verify: ;

pp07.run: run

