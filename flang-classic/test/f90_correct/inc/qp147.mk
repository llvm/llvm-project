#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test fold_const  ########


qp147: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp147.f08 fcheck.$(OBJX)
	-$(RM) qp147.$(EXESUFFIX) qp147.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp147.f08 -o qp147.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp147.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp147.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp147
	qp147.$(EXESUFFIX)

verify: ;

qp147.run: run

