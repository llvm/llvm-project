#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre29  ########


pre29: run
FFLAGS += -Mpreprocess

build:  $(SRC)/pre29.f90
	-$(RM) pre29.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre29.f90 -o pre29.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre29.$(OBJX) check.$(OBJX) $(LIBS) -o pre29.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre29
	pre29.$(EXESUFFIX)

verify: ;

pre29.run: run

