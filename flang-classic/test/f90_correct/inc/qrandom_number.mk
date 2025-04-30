#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qrandom_number  ########


qrandom_number: run
	

build:  $(SRC)/qrandom_number.f08
	-$(RM) qrandom_number.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qrandom_number.f08 -o qrandom_number.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qrandom_number.$(OBJX) check.$(OBJX) $(LIBS) -o qrandom_number.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qrandom_number
	qrandom_number.$(EXESUFFIX)

verify: ;

qrandom_number.run: run

