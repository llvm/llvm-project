#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre00  ########


pre00: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre00.f90
	-$(RM) pre00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre00.f90 -o pre00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre00.$(OBJX) check.$(OBJX) $(LIBS) -o pre00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre00
	pre00.$(EXESUFFIX)

verify: ;

pre00.run: run

