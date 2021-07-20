#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ParentMapContext.h"
#include <iostream>
#include <vector>

#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace llvm;
using namespace clang::tooling;
 

class ASTBuilderVisitor : public RecursiveASTVisitor<ASTBuilderVisitor>{
public:
    explicit ASTBuilderVisitor(ASTContext *Context) : Context(Context) {}

    bool VisitStmt(Stmt *s){
        const auto& parents = Context->getParents(*s);
        for(auto parent : parents){
            if(parent.get<Stmt>()){
                auto sPrime = parent.get<Stmt>();
                std::string parentStringRep;
                std::string stringRep;
                if(isa<BinaryOperator>(sPrime)){
                    if(cast<BinaryOperator>(sPrime)->isCommaOp()){
                        parentStringRep = "CommaOperator";
                    }
                    else{
                        parentStringRep = cast<BinaryOperator>(sPrime)->getOpcodeStr().str();
                    }
                }
                else if(isa<UnaryOperator>(sPrime)){
                    parentStringRep = cast<UnaryOperator>(sPrime)->getOpcodeStr(cast<UnaryOperator>(sPrime)->getOpcode()).str();
                }
                else{
                    std::string conversion1(sPrime->getStmtClassName());
                    parentStringRep = conversion1;
                }
                if(isa<BinaryOperator>(s)){
                    if(cast<BinaryOperator>(s)->isCommaOp()){
                        stringRep = "CommaOperator";
                    }
                    else{
                        stringRep =cast<BinaryOperator>(s)->getOpcodeStr().str();
                    }
                }
                else if(isa<UnaryOperator>(s)){
                    stringRep = cast<UnaryOperator>(s)->getOpcodeStr(cast<UnaryOperator>(s)->getOpcode()).str();
                }
                else{
                    std::string conversion2(s->getStmtClassName());
                    stringRep = conversion2;
                }
                std::cout <<"(AST,("<<sPrime<<","<<parentStringRep<<")"<<","<<"("<<s<<","<<stringRep<<"))"<<std::endl;
            }
            else if(parent.get<Decl>()){
                auto dPrime = parent.get<Decl>();
                std::cout <<"(AST,("<<dPrime<<","<<dPrime->getDeclKindName()<<")"<<","<<"("<<s<<","<<s->getStmtClassName()<<"))"<<std::endl;
            }
            else{
                errs()<<"I'm not sure what this is: " <<parent.getNodeKind().asStringRef().str()<<"\n";
            }
        }
        
        
        std::string inputString = "VERIFIER_nondet";
        if(isa<DeclRefExpr>(s)){
            DeclRefExpr* d = cast<DeclRefExpr>(s);
            if(d->getDecl()->getNameAsString().find(inputString)!=std::string::npos){
                std::cout<<"(AST,("<<s<<","<<s->getStmtClassName()<<")"<<","<<"("<<d->getDecl()<<","<<"input "<<d->getDecl()->getType().getAsString()<<"))"<<std::endl;
            }
            else{
                std::cout<<"(AST,("<<s<<","<<s->getStmtClassName()<<")"<<","<<"("<<d->getDecl()<<","<<d->getDecl()->getDeclKindName()<<"))"<<std::endl;
            }
        }

        return true;
    }

    bool VisitDecl(Decl *d){
        const auto& parents = Context->getParents(*d);
        std::string inputString = "VERIFIER_nondet";
        for(auto parent : parents){
            if(parent.get<Stmt>()){
                auto sPrime = parent.get<Stmt>();
                std::cout <<"(AST,("<<sPrime<<","<<sPrime->getStmtClassName()<<")"<<","<<"("<<d<<","<<d->getDeclKindName()<<"))"<<std::endl;
            }
            else if(parent.get<Decl>()){
                auto dPrime = parent.get<Decl>();
                if(isa<FunctionDecl>(d)){
                    FunctionDecl* f = cast<FunctionDecl>(d);
                    if(f->getNameInfo().getAsString().find(inputString) != std::string::npos){
                        std::cout <<"(AST,("<<dPrime<<","<<dPrime->getDeclKindName()<<")"<<","<<"("<<d<<","<<"input "<<f->getReturnType().getAsString()<<" ()))"<<std::endl;  
                    }
                    else{
                        std::cout <<"(AST,("<<dPrime<<","<<dPrime->getDeclKindName()<<")"<<","<<"("<<d<<","<<d->getDeclKindName()<<"))"<<std::endl;
                    }
                    for(auto param : f->parameters()){
                        std::cout<<"(AST,("<<d<<","<<d->getDeclKindName()<<")"<<","<<"("<<param<<","<<param->getDeclKindName()<<"))"<<std::endl;
                    }
                }
                else{
                    std::cout <<"(AST,("<<dPrime<<","<<dPrime->getDeclKindName()<<")"<<","<<"("<<d<<","<<d->getDeclKindName()<<"))"<<std::endl;
                }
            }
        }

        if(isa<VarDecl>(d)){
            VarDecl* v = cast<VarDecl>(d);

            if(v->getType().getTypePtr()->isBuiltinType()){
                    std::cout<<"(AST,("<<d<<","<<d->getDeclKindName()<<"),(typePlaceholder"<<placeholderVal++<<","<<v->getType().getDesugaredType(*Context).getAsString()<<"))"<<std::endl;
            }
            else if(v->getType().getTypePtr()->isStructureType()){
                std::cout<<"(AST,("<<d<<","<<d->getDeclKindName()<<")"<<","<<"(typePlaceholder"<<placeholderVal++<<",struct))"<<std::endl;
            }
            else if(v->getType().getTypePtr()->isArrayType()){
                std::cout<<"(AST,("<<d<<","<<d->getDeclKindName()<<")"<<","<<"(typePlaceholder"<<placeholderVal++<<",array))"<<std::endl;
            }
            else if(v->getType().getTypePtr()->isPointerType()){
                std::cout<<"(AST,("<<d<<","<<d->getDeclKindName()<<")"<<","<<"(typePlaceholder"<<placeholderVal++<<",pointer))"<<std::endl;
            }
            else{
                std::cout<<"(AST,("<<d<<","<<d->getDeclKindName()<<")"<<","<<"(typePlaceholder"<<placeholderVal++<<",otherType))"<<std::endl;
            }
        }

        return true;
    }

private:
    clang::ASTContext *Context;
    int placeholderVal = 0;
};

class CFGBuilderVisitor : public RecursiveASTVisitor<CFGBuilderVisitor>{
public:
    explicit CFGBuilderVisitor(ASTContext *Context) : Context(Context) {}
    
    /*
     * Takes in a Stmt representing the node we want to find the successor of
     * While a successor hasn't been found, look through the generation of ancestors
     * until a ancestor has been found with a sibling stmt that comes after it. Return sibling.
     * If ancestor is a FunctionDecl, that node must be the last stmt in the function, so return
     * an end function node.
     */
    DynTypedNode findSuccessor(const Stmt *node){
        auto parent = Context->getParents(*node).begin();
        bool found = false;
        if(parent->get<Stmt>()){
            for(auto child : parent->get<Stmt>()->children()){
                if(found){
                    return DynTypedNode::create(*child);
                }
                found = child == node;
            }
            return findSuccessor(parent->get<Stmt>());
        }
        else if(parent->get<Decl>()){
            return DynTypedNode::create(*(parent->get<Decl>()));
        }
        else{
            return DynTypedNode::create(*node);
        }     
    }

    void findReferencesHelper(const Stmt *orig, const DynTypedNode node){
        if(node.get<Stmt>()){
            for(auto child : node.get<Stmt>()->children()){
                if(child!=NULL){
                    if(isa<DeclRefExpr>(child)){
                        const DeclRefExpr *d = cast<DeclRefExpr>(child);
                        if(isa<VarDecl>(d->getDecl())){
                            std::cout<<"(Ref,"<<stmtNumber<<","<<orig<<",("<<d<<","<<d->getDecl()<<"))"<<std::endl;
                        }
                    }
                    else{
                        findReferencesHelper(orig, DynTypedNode::create(*child));
                    }
                }
            }
        }
    }

    void findReferences(const Stmt *node){
        //Find VarReference. I think thats all you need. Make sure it is not on the LHS of assignment or a ++/--
        if(node != NULL){
            if(isa<BinaryOperator>(node)){
                const BinaryOperator *b = cast<BinaryOperator>(node);
                if(b->isAssignmentOp()){
                    DynTypedNode d = DynTypedNode::create(*(b->getRHS()));
                    findReferencesHelper(node, d);
                }
                else{
                    DynTypedNode d = DynTypedNode::create(*node);
                    findReferencesHelper(node, d);
                }
            }
            else if (isa<CompoundStmt>(node)) {  }
            else{
                DynTypedNode d = DynTypedNode::create(*node);
                findReferencesHelper(node, d);
            }
        }
    }

    void printCFGPair(const Stmt* s1, const Stmt* s2){
        std::cout<<"(CFG,"<<stmtNumber<<",("<<s1<<","<<s1->getStmtClassName()<<"),("<<s2<<","<<s2->getStmtClassName()<<"))"<<std::endl;
        if(isa<DeclStmt>(s1)){
            const DeclStmt *d = cast<DeclStmt>(s1);
            for(auto dPrime : d->decls()){
                if(isa<VarDecl>(dPrime)){
                    VarDecl *v = cast<VarDecl>(dPrime);
                    std::cout<<"(Gen/Kill,"<<stmtNumber<<",("<<d<<","<<v<<"))"<<std::endl;
                }
            }
        }
        else if(isa<BinaryOperator>(s1)){
            const BinaryOperator *b = cast<BinaryOperator>(s1);
            if(b->isAssignmentOp()){
                if(isa<DeclRefExpr>(b->getLHS())){
                    DeclRefExpr *d = cast<DeclRefExpr>(b->getLHS());
                    ValueDecl *dPrime = d->getDecl();
                    std::cout<<"(Gen/Kill,"<<stmtNumber<<",("<<d<<","<<dPrime<<"))"<<std::endl;
                }
            }
        }
        else if(isa<UnaryOperator>(s1)){
            std::cout<<"You gotta handle a = ++b;"<<std::endl;
            const UnaryOperator *u = cast<UnaryOperator>(s1);
            if(u->isIncrementDecrementOp()){
                for(auto child : u->children()){
                    if(isa<DeclRefExpr>(child)){
                        const DeclRefExpr *d = cast<DeclRefExpr>(child);
                        const ValueDecl *dPrime = d->getDecl();
                        std::cout<<"(Gen/Kill,"<<stmtNumber<<",("<<d<<","<<dPrime<<")"<<std::endl;
                    }
                }
            } 
        }
        findReferences(s1);
        stmtNumber++;
    }
    
    void printCFGPair(const Stmt* s, const Decl* d){
        if(d){
            std::cout<<"(CFG,"<<stmtNumber<<",("<<s<<","<<s->getStmtClassName()<<"),("<<d<<","<<d->getDeclKindName()<<"))"<<std::endl;
        }
        else{
            return;
        }
        if(isa<DeclStmt>(s)){
            const DeclStmt *d = cast<DeclStmt>(s);
            for(auto dPrime : d->decls()){
                if(isa<VarDecl>(dPrime)){
                    VarDecl *v = cast<VarDecl>(dPrime);
                    std::cout<<"(Gen/Kill,"<<stmtNumber<<",("<<d<<","<<v<<"))"<<std::endl;
                }
            }
        }
        else if(isa<BinaryOperator>(s)){
            const BinaryOperator *b = cast<BinaryOperator>(s);
            if(b->isAssignmentOp()){
                if(isa<DeclRefExpr>(b->getLHS())){
                    DeclRefExpr *d = cast<DeclRefExpr>(b->getLHS());
                    ValueDecl *dPrime = d->getDecl();
                    std::cout<<"(Gen/Kill,"<<stmtNumber<<",("<<d<<","<<dPrime<<"))"<<std::endl;
                }
            }
        }
        else if(isa<UnaryOperator>(s)){
            const UnaryOperator *u = cast<UnaryOperator>(s);
            if(u->isIncrementDecrementOp()){
                for(auto child : u->children()){
                    if(isa<DeclRefExpr>(child)){
                        const DeclRefExpr *d = cast<DeclRefExpr>(child);
                        const ValueDecl *dPrime = d->getDecl();
                        std::cout<<"(Gen/Kill,"<<stmtNumber<<",("<<d<<","<<dPrime<<")"<<std::endl;
                    }
                }
            }
        }
        findReferences(s);
        stmtNumber++;
    }

    void printCFGPair(const Decl* d, const Stmt* s){
        std::cout<<"(CFG,("<<d<<","<<d->getDeclKindName()<<"),("<<s<<","<<s->getStmtClassName()<<"))"<<std::endl;
        stmtNumber++;
    }

    void printCFGPair(const Decl* d1, const Decl* d2){
        std::cout<<"(CFG,("<<d1<<","<<d1->getDeclKindName()<<"),("<<d2<<","<<d2->getDeclKindName()<<"))"<<std::endl;
        stmtNumber++;
    }

    /*Links all items in a compound statement in order of execution*/
    bool VisitCompoundStmt(CompoundStmt* cmpdStmt){
        Stmt *prevChild = cmpdStmt;
        
        for(auto child : cmpdStmt->children()){
            bool isTakenCareOf = isa<IfStmt>(prevChild) || isa<BreakStmt>(prevChild) || WhileStmt::classof(prevChild) || 
                                 isa<ForStmt>(prevChild) || isa<DoStmt>(prevChild) || isa<ContinueStmt>(prevChild) ||
                                 SwitchStmt::classof(prevChild) || DefaultStmt::classof(prevChild);
            if(isTakenCareOf){}
            else{
                printCFGPair(prevChild, child);
                prevChild = child;
            }
        }
        
        return true;
    }

    bool VisitCallExpr(CallExpr *call){
        Decl* funDecl = call->getCalleeDecl();
        printCFGPair(call, funDecl);

        auto successor = findSuccessor(call);

        if(funDecl){
            if(successor.get<Stmt>()){
                std::cout<<"(CFG,"<<stmtNumber<<",("<<funDecl<<",FunctionExit),("<<successor.get<Stmt>()<<","<<successor.get<Stmt>()->getStmtClassName()<<")"<<std::endl;
                stmtNumber++;
            }
            else if(successor.get<FunctionDecl>()){
                std::cout<<"(CFG,"<<stmtNumber<<",("<<funDecl<<",FunctionExit),("<<successor.get<FunctionDecl>()<<",FunctionExit))"<<std::endl;
                stmtNumber++;
            }
            else if(successor.get<Decl>()){
                std::cout<<"(CFG,"<<stmtNumber<<",("<<funDecl<<",FunctionExit),("<<successor.get<Decl>()<<","<<successor.get<Decl>()->getDeclKindName()<<"))"<<std::endl;
                stmtNumber++;
            }
        }
        return true;
    }

    bool VisitIfStmt(IfStmt *i){
        Stmt* thenStmt = i->getThen();

        DynTypedNode successor = findSuccessor(i);

        /* Link If header to then and else branch if they exist */
        if(thenStmt){
            printCFGPair(i, thenStmt);

            auto finalNode = thenStmt;
            if(isa<CompoundStmt>(thenStmt)){
                for(auto thenChild : thenStmt->children()){
                    finalNode = thenChild;
                }
            }
            if(successor.get<Stmt>()){ 
                printCFGPair(finalNode, successor.get<Stmt>());
            }
            else if(successor.get<FunctionDecl>()){
                std::cout<<"(CFG,"<<stmtNumber<<",("<<finalNode<<","<<finalNode->getStmtClassName()<<"),("<<successor.get<FunctionDecl>()<<",FunctionExit))"<<std::endl;
                findReferences(finalNode);
                stmtNumber++;
            }
            else if(successor.get<Decl>()){
                printCFGPair(finalNode, successor.get<Decl>());
            }
        }

        Stmt* elseStmt = i->getElse();
        if(elseStmt){
            printCFGPair(i, elseStmt);

            auto finalNode = elseStmt;
            if(isa<CompoundStmt>(elseStmt)){
                for(auto elseChild : thenStmt->children()){
                    finalNode = elseChild;
                }
            }
            if(successor.get<Stmt>()){ 
                printCFGPair(finalNode, successor.get<Stmt>());
            }
            else if(successor.get<FunctionDecl>()){
                std::cout<<"(CFG,"<<stmtNumber<<",("<<finalNode<<","<<finalNode->getStmtClassName()<<"),("<<successor.get<FunctionDecl>()<<",FunctionExit))"<<std::endl;
                findReferences(finalNode);
                stmtNumber++;
            }
            else if(successor.get<Decl>()){
                printCFGPair(finalNode, successor.get<Decl>());
            }
        }
        else{
            if(successor.get<Stmt>()){
                printCFGPair(i, successor.get<Stmt>());
            }
            else if(successor.get<FunctionDecl>()){
                std::cout<<"(CFG,"<<stmtNumber<<",("<<i<<","<<i->getStmtClassName()<<"),("<<successor.get<FunctionDecl>()<<",FunctionExit))"<<std::endl;
                findReferences(i);
                stmtNumber++;
            }
            else if(successor.get<Decl>()){
                printCFGPair(i, successor.get<Decl>());
            }
        }
        return true;
    }

    bool VisitSwitchStmt(SwitchStmt *s){
        
        auto nextCase = s->getSwitchCaseList();
        while(nextCase){
            printCFGPair(s, nextCase);
            nextCase = nextCase->getNextSwitchCase();
        }

        return true;
    }

    bool VisitCaseStmt(CaseStmt *c){
        auto body = c->getSubStmt();

        printCFGPair(c, body);
        return true;
    }

    bool VisitReturnStmt(ReturnStmt *r){
        bool foundFunction = false;
        bool stopCondition = false;

        auto parent = Context->getParents(*r).begin();
        while(!stopCondition){
            foundFunction = parent->get<FunctionDecl>();
            stopCondition = (foundFunction || parent->get<TranslationUnitDecl>());
            if(stopCondition) break;

            if(parent->get<Stmt>()){
                parent = Context->getParents(*(parent->get<Stmt>())).begin();
            }
            else if(parent->get<Decl>()){
                parent = Context->getParents(*(parent->get<Decl>())).begin();
            }
        }

        if(parent->get<FunctionDecl>()){
            std::cout<<"(CFG,"<<stmtNumber<<",("<<r<<","<<r->getStmtClassName()<<"),("<<parent->get<FunctionDecl>()<<",FunctionExit))"<<std::endl;
            findReferences(r);
            stmtNumber++;
        }

        return true;
    }

    bool VisitDefaultStmt(DefaultStmt *d){
        auto body = d->getSubStmt();

        printCFGPair(d, body);

        auto successor = findSuccessor(Context->getParents(*d).begin()->get<Stmt>());

        auto finalNode = body;
        if(isa<CompoundStmt>(body)){
            for(auto bodyChild : body->children()){
                finalNode = bodyChild;
            }
        }

        if(successor.get<Stmt>()){
            printCFGPair(finalNode, successor.get<Stmt>());
        }
        else if(successor.get<FunctionDecl>()){
            std::cout<<"(CFG,"<<stmtNumber<<",("<<finalNode<<","<<finalNode->getStmtClassName()<<"),("<<successor.get<FunctionDecl>()<<",FunctionExit))"<<std::endl;
            findReferences(finalNode);
            stmtNumber++;
        }
        else if(successor.get<Decl>()){
            printCFGPair(finalNode, successor.get<Decl>());
        }
        return true;
    }

    bool VisitWhileStmt(WhileStmt *w){
        Stmt *body = w->getBody();
        
        //header to body
        printCFGPair(w, body);

        auto successor = findSuccessor(w);

        auto finalNode = body;
        if(isa<CompoundStmt>(body)){
            for(auto bodyChild : body->children()){
                finalNode = bodyChild;
            }
        }

        //body back to header
        printCFGPair(finalNode, w);

        //header to next node
        if(successor.get<Stmt>()){ 
            printCFGPair(w, successor.get<Stmt>());
        }
        else if(successor.get<FunctionDecl>()){
            std::cout<<"(CFG,"<<stmtNumber<<",("<<w<<","<<w->getStmtClassName()<<"),("<<successor.get<FunctionDecl>()<<",FunctionExit))"<<std::endl;
            findReferences(w);
            stmtNumber++;
        }
        else if(successor.get<Decl>()){
            printCFGPair(w, successor.get<Decl>());
        }
        return true;
    }

    bool VisitDoStmt(DoStmt *d){
        Stmt *body = d->getBody();
        
        //header to body
        printCFGPair(d, body);

        auto successor = findSuccessor(d);

        auto finalNode = body;
        if(isa<CompoundStmt>(body)){
            for(auto bodyChild : body->children()){
                finalNode = bodyChild;
            }
        }

        //body back to header
        printCFGPair(finalNode, d);

        //header to next node
        if(successor.get<Stmt>()){ 
            printCFGPair(d, successor.get<Stmt>());
        }
        else if(successor.get<FunctionDecl>()){
            std::cout<<"(CFG,"<<stmtNumber<<",("<<d<<","<<d->getStmtClassName()<<"),("<<successor.get<FunctionDecl>()<<",FunctionExit))"<<std::endl;
            findReferences(d);
            stmtNumber++;
        }
        else if(successor.get<Decl>()){
            printCFGPair(d, successor.get<Decl>());
        }

        return true;
    }

    bool VisitForStmt(ForStmt *f){
        Stmt* body = f->getBody();
        printCFGPair(f, body);

        auto successor = findSuccessor(f);

        auto finalNode = body;
        if(isa<CompoundStmt>(body)){
            for(auto bodyChild : body->children()){
                finalNode = bodyChild;
            }
        }

        //body back to header
        printCFGPair(finalNode, f);

        //header to next node
        if(successor.get<Stmt>()){ 
            printCFGPair(f, successor.get<Stmt>());
        }
        else if(successor.get<FunctionDecl>()){
            std::cout<<"(CFG,"<<stmtNumber<<",("<<f<<","<<f->getStmtClassName()<<"),("<<successor.get<FunctionDecl>()<<",FunctionExit))"<<std::endl;
            findReferences(f);
            stmtNumber++;
        }
        else if(successor.get<Decl>()){
            printCFGPair(f, successor.get<Decl>());
        }

        return true;
    }

    bool VisitGotoStmt(GotoStmt *g){
        printCFGPair(g, g->getLabel()->getStmt());

        return true;
    }

    bool VisitBreakStmt(BreakStmt *b){
        bool parentIsSwitch = false;
        bool parentIsWhile = false;
        bool parentIsCase = false;
        bool parentIsFor = false;
        bool parentIsDo = false;

        bool stopCondition = false;
        auto parent = Context->getParents(*b).begin();
        while(!stopCondition){
            parentIsSwitch = parent->get<Stmt>()->getStmtClass() == SwitchStmt::CreateEmpty(*Context, false, false)->getStmtClass();
            parentIsWhile = parent->get<WhileStmt>();
            parentIsCase = parent->get<CaseStmt>();
            parentIsFor = parent->get<ForStmt>();
            parentIsDo = parent->get<DoStmt>();

            stopCondition = (parentIsSwitch || parentIsCase || parentIsWhile || parentIsFor || parentIsDo || parent->get<TranslationUnitDecl>());
            
            if(parent->get<Stmt>()){
                parent = Context->getParents(*(parent->get<Stmt>())).begin();
            }
            else if(parent->get<Decl>()){
                parent = Context->getParents(*(parent->get<Decl>())).begin();
            }
        }
        if(parent->get<Stmt>()){
            auto successor = findSuccessor(parent->get<Stmt>());
            if(successor.get<Stmt>()){ 
                printCFGPair(b, successor.get<Stmt>());
            }
            else if(successor.get<FunctionDecl>()){
                std::cout<<"(CFG,"<<stmtNumber<<",("<<b<<","<<b->getStmtClassName()<<"),("<<successor.get<FunctionDecl>()<<",FunctionExit))"<<std::endl;
                stmtNumber++;
            }
            else if(successor.get<Decl>()){
                printCFGPair(b, successor.get<Decl>());
            }
        }
        return true;
    }

    bool VisitContinueStmt(ContinueStmt *c){
        bool parentIsWhile = false;
        bool parentIsFor = false;
        bool parentIsDo = false;

        bool stopCondition = false;
        auto parent = Context->getParents(*c).begin();
        while(!stopCondition){
            parentIsWhile = parent->get<WhileStmt>();
            parentIsFor = parent->get<ForStmt>();
            parentIsDo = parent->get<DoStmt>();

            stopCondition = (parentIsWhile || parentIsFor || parentIsDo ||  parent->get<TranslationUnitDecl>());
            if(stopCondition) break;

            if(parent->get<Stmt>()){
                parent = Context->getParents(*(parent->get<Stmt>())).begin();
            }
            else if(parent->get<Decl>()){
                parent = Context->getParents(*(parent->get<Decl>())).begin();
            }
        }
        if(parent->get<Stmt>()){
            printCFGPair(c, parent->get<Stmt>());
        }

        return true;
    }

    bool VisitDecl(Decl *d){
        if(isa<FunctionDecl>(d)){
            FunctionDecl* f = cast<FunctionDecl>(d);
            std::string stringRep = "Function";
            if(f->hasBody()){
                auto functionHead = f->getBody();
                if(f->getNameInfo().getAsString() == "main"){
                    stringRep = "main";
                }
                std::cout<<"(CFG,"<<stmtNumber<<",("<<f<<","<<stringRep<<"),("<<functionHead<<","<<functionHead->getStmtClassName()<<"))"<<std::endl;
                stmtNumber++;
            }
        }
        return true;
    }
private:
    clang::ASTContext *Context;
    int stmtNumber = 0;
};

class DFGBuilderVisitor : public RecursiveASTVisitor<DFGBuilderVisitor>{
public:
    explicit DFGBuilderVisitor(ASTContext *Context) : Context(Context) {}

    void printDFGPair(const Stmt* s1, const Stmt* s2){
        std::cout<<"(DFG,("<<s1<<","<<s1->getStmtClassName()<<"),("<<s2<<","<<s2->getStmtClassName()<<"))"<<std::endl;
    }
    
    void printDFGPair(const Stmt* s, const Decl* d){
        std::cout<<"(DFG,("<<s<<","<<s->getStmtClassName()<<"),("<<d<<","<<d->getDeclKindName()<<"))"<<std::endl;
    }

    void printDFGPair(const Decl* d, const Stmt* s){
        std::cout<<"(DFG,("<<d<<","<<d->getDeclKindName()<<"),("<<s<<","<<s->getStmtClassName()<<"))"<<std::endl;
    }

    void printDFGPair(const Decl* d1, const Decl* d2){
        std::cout<<"(DFG,("<<d1<<","<<d1->getDeclKindName()<<"),("<<d2<<","<<d2->getDeclKindName()<<"))"<<std::endl;
    }

    DeclRefExpr * findDeclRefExpr(Stmt *s){
        if(isa<DeclRefExpr>(s)){
            return cast<DeclRefExpr>(s);
        }
        else{
            for(auto child : s->children()){
                return findDeclRefExpr(child);
            }
            return nullptr;
        }
    }

    void findReturnStmt(Decl *lhs, Stmt *s){
        if(s){
            if(isa<ReturnStmt>(s)){
                std::cout<<"(DFG,("<<lhs<<","<<lhs->getDeclKindName()<<"),("<<s<<","<<s->getStmtClassName()<<"))"<<std::endl;
                for(auto child : s->children()){
                    binaryOpHelper(s, child);
                }
            }
            else{
                for(auto child : s->children()){
                    findReturnStmt(lhs, child);
                }
            }
        }
    }

    void binaryOpHelper(Stmt *lhs, Stmt *child){
        if(child){
            if(isa<CallExpr>(child)){  }
            else{
                printDFGPair(child, lhs);
                if(!(child->children().empty())){
                    for(auto childPrime : child->children()){
                        binaryOpHelper(lhs, childPrime);
                    }
                }
            }
        }
    }


    void binaryOpHelper(Decl *lhs, Stmt *child){
        if(child){
            if(isa<CallExpr>(child)){
                
                auto call = cast<CallExpr>(child); 
                if(call->getCalleeDecl()){
                    if(isa<FunctionDecl>(call->getCalleeDecl())){
                        
                        auto fd = cast<FunctionDecl>(call->getCalleeDecl()); 
                        if(fd->hasBody()){
                            findReturnStmt(lhs, fd->getBody());
                        }
                        else{
                            if(fd->getNameAsString().find("__VERIFIER_nondet")!=std::string::npos){
                                std::cout<<"(DFG,("<<lhs<<","<<lhs->getDeclKindName()<<"),("<<fd<<","<<"input "<<fd->getType().getAsString()<<"))"<<std::endl;
                            }
                            else{
                                std::cout<<"(DFG,("<<lhs<<","<<lhs->getDeclKindName()<<"),("<<fd<<","<<fd->getType().getAsString()<<"))"<<std::endl;
                            }
                        }
                    }
                }
            }
            else{
                printDFGPair(child, lhs);
                if(!(child->children().empty())){
                    for(auto childPrime : child->children()){
                        binaryOpHelper(lhs, childPrime);
                    }
                }
            }
        }
    }

    bool VisitBinaryOperator(BinaryOperator *b){
        if(b->isCompoundAssignmentOp()){
            if(isa<DeclRefExpr>(b->getLHS())){
                
                auto c = cast<DeclRefExpr>(b->getLHS());
                printDFGPair(c->getDecl(), c->getDecl());
                binaryOpHelper(c->getDecl(), b->getRHS());
            }
            
        }
        else if(b->isAssignmentOp()){
            if(isa<DeclRefExpr>(b->getLHS())){
                
                auto c = cast<DeclRefExpr>(b->getLHS());
                binaryOpHelper(c->getDecl(), b->getRHS());
            }
            else{
                binaryOpHelper(b->getLHS(), b->getRHS());
            }
        }

        return true;
    }

    bool VisitUnaryOperator(UnaryOperator *u){
        if(u->isIncrementOp() || u->isDecrementOp()){
            for(auto child : u->children()){
                DeclRefExpr *d = findDeclRefExpr(child);
                if(d != NULL){
                    printDFGPair(d->getDecl(), d->getDecl());
                }
            }
        }
        return true;
    }

    bool VisitVarDecl(VarDecl *v){
        if(v->hasInit()){
            binaryOpHelper(v, v->getInit());
        }
        return true;
    }

    bool VisitCallExpr(CallExpr *c){
        auto d1 = c->getCalleeDecl();
        
        if(!isa<FunctionDecl>(d1)){
            return true;
        }
        auto d2 = cast<FunctionDecl>(d1);
        for(auto [param, arg] : zip(d2->parameters(),c->arguments())){
            printDFGPair(arg, param);
        }

        return true;
    }

private:
    clang::ASTContext *Context;
};

class GraphBuilderConsumer : public clang::ASTConsumer {
public:
    explicit GraphBuilderConsumer(ASTContext *Context) : VisitorAST(Context), VisitorCFG(Context), VisitorDFG(Context){}
    virtual void HandleTranslationUnit(clang::ASTContext &Context) override {
        VisitorAST.TraverseDecl(Context.getTranslationUnitDecl());
        VisitorCFG.TraverseDecl(Context.getTranslationUnitDecl());
        VisitorDFG.TraverseDecl(Context.getTranslationUnitDecl());
        
    }

private:
    ASTBuilderVisitor VisitorAST;
    CFGBuilderVisitor VisitorCFG;
    DFGBuilderVisitor VisitorDFG;
};

class GraphBuilderAction : public clang::ASTFrontendAction {
public:
    virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
        clang::CompilerInstance &Compiler, llvm::StringRef InFile) override{
        return std::unique_ptr<clang::ASTConsumer>(
            new GraphBuilderConsumer(&Compiler.getASTContext()));
        }
};


static cl::OptionCategory MyToolCategory("My tool options");
int main(int argc, const char **argv) {
  CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());
  return Tool.run(newFrontendActionFactory<GraphBuilderAction>().get());
}