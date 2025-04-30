#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre01  ########


pre01: run
FFLAGS += -Mpreprocess
	

build:  $(SRC)/pre01.f90
	-$(RM) pre01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre01.f90 -o pre01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre01.$(OBJX) check.$(OBJX) $(LIBS) -o pre01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre01
	pre01.$(EXESUFFIX)

verify: ;

pre01.run: run

