#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test cb00  ########


cb00: run
	

build:  $(SRC)/cb00.f
	-$(RM) cb00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/cb00.f -o cb00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) cb00.$(OBJX) check.$(OBJX) $(LIBS) -o cb00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test cb00
	cb00.$(EXESUFFIX)

verify: ;

cb00.run: run

