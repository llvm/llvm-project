#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka39  ########


ka39: run
	

build:  $(SRC)/ka39.f
	-$(RM) ka39.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka39.f -o ka39.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka39.$(OBJX) check.$(OBJX) $(LIBS) -o ka39.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka39
	ka39.$(EXESUFFIX)

verify: ;

ka39.run: run

