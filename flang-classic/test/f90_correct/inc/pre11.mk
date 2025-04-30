#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre11  ########


pre11: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre11.f90
	-$(RM) pre11.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre11.f90 -o pre11.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre11.$(OBJX) check.$(OBJX) $(LIBS) -o pre11.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre11
	pre11.$(EXESUFFIX)

verify: ;

pre11.run: run

