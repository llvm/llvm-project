#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test pre30  ########


pre30: run
FFLAGS += -mp -Mpreprocess
	

build:  $(SRC)/pre30.f
	-$(RM) pre30.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/pre30.f -o pre30.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) pre30.$(OBJX) check.$(OBJX) $(LIBS) -o pre30.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test pre30
	pre30.$(EXESUFFIX)

verify: ;

pre30.run: run

