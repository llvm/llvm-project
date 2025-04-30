#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre15  ########


pre15: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre15.f90
	-$(RM) pre15.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre15.f90 -o pre15.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre15.$(OBJX) check.$(OBJX) $(LIBS) -o pre15.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre15
	pre15.$(EXESUFFIX)

verify: ;

pre15.run: run

