#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre36  ########


pre36: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre36.f
	-$(RM) pre36.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre36.f -o pre36.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre36.$(OBJX) check.$(OBJX) $(LIBS) -o pre36.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre36
	pre36.$(EXESUFFIX)

verify: ;

pre36.run: run

