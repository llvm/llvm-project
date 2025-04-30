#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre14  ########


pre14: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre14.f90
	-$(RM) pre14.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre14.f90 -o pre14.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre14.$(OBJX) check.$(OBJX) $(LIBS) -o pre14.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre14
	pre14.$(EXESUFFIX)

verify: ;

pre14.run: run

