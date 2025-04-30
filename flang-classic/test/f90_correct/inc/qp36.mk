#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp36  ########


qp36: run


build:  $(SRC)/qp36.f08
	-$(RM) qp36.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp36.f08 -o qp36.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp36.$(OBJX) check.$(OBJX) $(LIBS) -o qp36.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp36
	qp36.$(EXESUFFIX)

verify: ;
