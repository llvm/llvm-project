#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka51  ########


ka51: run
	

build:  $(SRC)/ka51.f
	-$(RM) ka51.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka51.f -o ka51.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka51.$(OBJX) check.$(OBJX) $(LIBS) -o ka51.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka51
	ka51.$(EXESUFFIX)

verify: ;

ka51.run: run

