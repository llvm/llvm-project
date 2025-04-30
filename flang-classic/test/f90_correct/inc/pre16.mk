#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre16  ########


pre16: run
FFLAGS += -Mpreprocess -Hx,124,0x100000 # Do not expand comments

build:  $(SRC)/pre16.f90
	-$(RM) pre16.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre16.f90 -o pre16.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre16.$(OBJX) check.$(OBJX) $(LIBS) -o pre16.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre16
	pre16.$(EXESUFFIX)

verify: ;

pre16.run: run

