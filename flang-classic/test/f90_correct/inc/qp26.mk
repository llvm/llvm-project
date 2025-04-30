#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp26  ########


qp26: run


build:  $(SRC)/qp26.f08
	-$(RM) qp26.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp26.f08 -o qp26.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp26.$(OBJX) check.$(OBJX) $(LIBS) -o qp26.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp26
	qp26.$(EXESUFFIX)

verify: ;
