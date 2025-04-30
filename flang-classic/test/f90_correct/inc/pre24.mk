#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre24  ########


pre24: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre24.f
	-$(RM) pre24.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre24.f -o pre24.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre24.$(OBJX) check.$(OBJX) $(LIBS) -o pre24.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre24
	pre24.$(EXESUFFIX)

verify: ;

pre24.run: run

