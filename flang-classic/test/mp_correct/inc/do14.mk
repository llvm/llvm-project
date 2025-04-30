#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
do14: do14.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
do14.$(OBJX): $(SRC)/do14.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/do14.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) do14.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: do14
run: ;
