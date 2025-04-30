#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pp52  ########


pp52: run
	

build:  $(SRC)/pp52.f90
	-$(RM) pp52.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pp52.f90 -o pp52.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pp52.$(OBJX) check.$(OBJX) $(LIBS) -o pp52.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pp52
	pp52.$(EXESUFFIX)

verify: ;

pp52.run: run

