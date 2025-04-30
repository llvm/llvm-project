#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pp03  ########


pp03: run
	

build:  $(SRC)/pp03.f90
	-$(RM) pp03.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pp03.f90 -o pp03.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pp03.$(OBJX) check.$(OBJX) $(LIBS) -o pp03.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pp03
	pp03.$(EXESUFFIX)

verify: ;

pp03.run: run

