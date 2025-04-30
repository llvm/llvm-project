#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre12  ########


pre12: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre12.f90
	-$(RM) pre12.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre12.f90 -o pre12.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre12.$(OBJX) check.$(OBJX) $(LIBS) -o pre12.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre12
	pre12.$(EXESUFFIX)

verify: ;

pre12.run: run

