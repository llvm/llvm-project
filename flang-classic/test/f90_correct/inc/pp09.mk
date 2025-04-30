#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pp09  ########


pp09: run
	

build:  $(SRC)/pp09.f90
	-$(RM) pp09.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pp09.f90 -o pp09.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pp09.$(OBJX) check.$(OBJX) $(LIBS) -o pp09.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pp09
	pp09.$(EXESUFFIX)

verify: ;

pp09.run: run

