// Append using posix-style a file name or directory to Base
function append(Base, New) {
  if (!New)
    return Base;
  if (Base)
    Base += "/";
  Base += New;
  return Base;
}

// Get relative path to access FilePath from CurrentDirectory
function computeRelativePath(FilePath, CurrentDirectory) {
  var Path = FilePath;
  while (Path) {
    if (CurrentDirectory == Path)
      return FilePath.substring(Path.length + 1);
    Path = Path.substring(0, Path.lastIndexOf("/"));
  }

  var Dir = CurrentDirectory;
  var Result = "";
  while (Dir) {
    if (Dir == FilePath)
      break;
    Dir = Dir.substring(0, Dir.lastIndexOf("/"));
    Result = append(Result, "..")
  }
  Result = append(Result, FilePath.substring(Dir.length))
  return Result;
}

function genLink(Ref, CurrentDirectory) {
  var Path = computeRelativePath(Ref.Path, CurrentDirectory);
  Path = append(Path, Ref.Name + ".html")
  ANode = document.createElement("a");
  ANode.setAttribute("href", Path);
  var TextNode = document.createTextNode(Ref.Name);
  ANode.appendChild(TextNode);
  return ANode;
}

function genHTMLOfIndex(Index, CurrentDirectory, IsOutermostList) {
  // Out will store the HTML elements that Index requires to be generated
  var Out = [];
  if (Index.Name) {
    var SpanNode = document.createElement("span");
    var TextNode = document.createTextNode(Index.Name);
    SpanNode.appendChild(genLink(Index, CurrentDirectory));
    Out.push(SpanNode);
  }
  if (Index.Children.length == 0)
    return Out;
  // Only the outermost list should use ol, the others should use ul
  var ListNodeName = IsOutermostList ? "ol" : "ul";
  var ListNode = document.createElement(ListNodeName);
  for (Child of Index.Children) {
    var LiNode = document.createElement("li");
    ChildNodes = genHTMLOfIndex(Child, CurrentDirectory, false);
    for (Node of ChildNodes)
      LiNode.appendChild(Node);
    ListNode.appendChild(LiNode);
  }
  Out.push(ListNode);
  return Out;
}

function createIndex(Index) {
  // Get the DOM element where the index will be created
  var IndexDiv = document.getElementById("sidebar-left");
  // Get the relative path of this file
  CurrentDirectory = IndexDiv.getAttribute("path");
  var IndexNodes = genHTMLOfIndex(Index, CurrentDirectory, true);
  for (Node of IndexNodes)
    IndexDiv.appendChild(Node);
}

// Runs after DOM loads
document.addEventListener("DOMContentLoaded", function() {
  // JsonIndex is a variable from another file that contains the index
  // in JSON format
  var Index = JSON.parse(JsonIndex);
  createIndex(Index);
});
